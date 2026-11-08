import gc
from pathlib import Path

import numpy as np
import torch

import utils
from models import SynthesizerTrn


class F0FilterException(Exception):
    pass


class Svc:
    """Whisper large-v3 content features + the So-VITS-SVC generator."""

    def __init__(self, net_g_path, config_path, device=None, precision=None, encoder_precision=None,
                 nsf_hifigan_enhance=False):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        precision = precision or ("fp16" if self.dev.type == "cuda" else "fp32")
        if precision not in {"fp16", "fp32"}:
            raise ValueError("Generator precision must be fp16 or fp32")
        if self.dev.type != "cuda" and precision != "fp32":
            raise ValueError("CPU inference requires fp32 precision")
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.hps = utils.get_hparams_from_file(config_path)
        self.target_sample = self.hps.data.sampling_rate
        self.hop_size = self.hps.data.hop_length
        self.spk2id = self.hps.spk
        self.unit_interpolate_mode = self.hps.data.unit_interpolate_mode
        self.vol_embedding = self.hps.model.vol_embedding
        self.net_g_ms = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hop_size,
            **self.hps.model,
        )
        checkpoint = torch.load(net_g_path, map_location="cpu", weights_only=True, mmap=True)
        self.net_g_ms.load_state_dict(checkpoint["model"], strict=True)
        del self.net_g_ms.enc_q
        self.net_g_ms.eval().requires_grad_(False).to(device=self.dev, dtype=self.dtype)
        self.hubert_model = utils.get_speech_encoder(
            self.hps.model.speech_encoder, device=self.dev, precision=encoder_precision,
        )
        self.volume_extractor = utils.Volume_Extractor(self.hop_size)
        self.f0_predictor_object = None
        self.enhancer = None
        if nsf_hifigan_enhance:
            from modules.enhancer import Enhancer
            self.enhancer = Enhancer("nsf-hifigan", "pretrain/nsf_hifigan/model", device=self.dev)

    def _speaker_id(self, speaker):
        if isinstance(speaker, int) and 0 <= speaker < len(self.spk2id):
            return speaker
        speaker_id = self.spk2id.get(speaker)
        if speaker_id is None:
            raise ValueError(f"Unknown speaker {speaker!r}; choices: {list(self.spk2id.keys())}")
        return speaker_id

    @torch.inference_mode()
    def infer(self, speaker, tran, raw_path, auto_predict_f0=False, noice_scale=0.4,
              f0_filter=False, f0_predictor="pm", loudness_envelope_adjustment=1,
              f0_filter_threshold=0.05, enhancer_adaptive_key=0):
        import librosa

        speaker_id = self._speaker_id(speaker)
        if isinstance(raw_path, (str, Path)) or hasattr(raw_path, "read"):
            wav, _ = librosa.load(raw_path, sr=self.target_sample, mono=True)
        else:
            wav = np.asarray(raw_path, dtype=np.float32)
        if (self.f0_predictor_object is None or self.f0_predictor_object.name != f0_predictor
                or (f0_predictor in {"crepe", "rmvpe", "fcpe"}
                    and self.f0_predictor_object.threshold != f0_filter_threshold)):
            self.f0_predictor_object = utils.get_f0_predictor(
                f0_predictor, hop_length=self.hop_size, sampling_rate=self.target_sample,
                device=self.dev, threshold=f0_filter_threshold,
            )
        f0, uv = self.f0_predictor_object.compute_f0_uv(wav)
        if f0_filter and not np.any(f0):
            raise F0FilterException("No voice detected")
        wav16k = librosa.resample(wav, orig_sr=self.target_sample, target_sr=16000)
        c = self.hubert_model.encoder(torch.from_numpy(wav16k).to(self.dev))
        c = utils.repeat_expand_2d(c.squeeze(0), len(f0), self.unit_interpolate_mode)
        c = c.unsqueeze(0).to(device=self.dev, dtype=self.dtype)
        f0 = torch.as_tensor(f0, device=self.dev, dtype=self.dtype)[None] * 2 ** (tran / 12)
        uv = torch.as_tensor(uv, device=self.dev, dtype=self.dtype)[None]
        sid = torch.tensor([[speaker_id]], device=self.dev)
        vol = None
        if self.vol_embedding:
            vol = self.volume_extractor.extract(torch.from_numpy(wav)[None].to(self.dev))
            vol = vol[None, :f0.shape[1]].to(dtype=self.dtype)
        audio, generated_f0 = self.net_g_ms.infer(
            c, f0=f0, uv=uv, g=sid, predict_f0=auto_predict_f0,
            noice_scale=noice_scale, vol=vol,
        )
        audio = audio[0, 0].float()
        if self.enhancer is not None:
            audio, sample_rate = self.enhancer.enhance(
                audio[None], self.target_sample, generated_f0[:, :, None], self.hop_size,
                adaptive_key=enhancer_adaptive_key,
            )
            if sample_rate != self.target_sample:
                from modules.F0Predictor.audio import resample_audio
                audio = resample_audio(audio, sample_rate, self.target_sample)
        if loudness_envelope_adjustment != 1:
            audio = utils.change_rms(
                wav, self.target_sample, audio, self.target_sample, loudness_envelope_adjustment,
            )
        return audio, audio.numel(), f0.shape[1]

    def slice_inference(self, raw_audio_path, spk, tran=0, slice_db=-40,
                        auto_predict_f0=False, noice_scale=0.4, pad_seconds=0.5,
                        clip_seconds=0, lg_num=0, lgr_num=0.75, f0_predictor="pm",
                        loudness_envelope_adjustment=1, f0_filter_threshold=0.05,
                        enhancer_adaptive_key=0):
        import librosa
        from inference.slicer import Slicer

        self._speaker_id(spk)
        if not 0 <= pad_seconds < 15 or clip_seconds < 0 or lg_num < 0 or not 0 < lgr_num <= 1:
            raise ValueError("Invalid padding, clipping or crossfade settings")
        wav, _ = librosa.load(raw_audio_path, sr=self.target_sample, mono=True)
        if wav.size == 0:
            raise ValueError("Input audio is empty")
        # Whisper only accepts 30 seconds, including the padding on both sides.
        max_seconds = 30 - 2 * pad_seconds
        window = int(min(clip_seconds or max_seconds, max_seconds) * self.target_sample)
        overlap = int(lg_num * self.target_sample)
        if window < self.hop_size or overlap >= window:
            raise ValueError("Clip must be at least one frame and crossfade shorter than the clip")
        pad = int(pad_seconds * self.target_sample)
        output = np.zeros(wav.size, dtype=np.float32)
        chunks = Slicer(self.target_sample, threshold=slice_db).slice(wav)
        for chunk in chunks.values():
            begin, end = map(int, chunk["split_time"].split(","))
            if chunk["slice"] or begin == end:
                continue
            position = begin
            while position < end:
                stop = min(position + window, end)
                padded = np.pad(wav[position:stop], (pad, pad))
                audio, _, _ = self.infer(
                    spk, tran, padded, auto_predict_f0=auto_predict_f0,
                    noice_scale=noice_scale, f0_predictor=f0_predictor,
                    f0_filter_threshold=f0_filter_threshold,
                    enhancer_adaptive_key=enhancer_adaptive_key,
                    loudness_envelope_adjustment=loudness_envelope_adjustment,
                )
                converted = audio.cpu().numpy()
                if pad:
                    converted = converted[pad:-pad]
                length = stop - position
                if converted.size < length:
                    missing = length - converted.size
                    converted = np.pad(converted, (missing // 2, missing - missing // 2))
                converted = converted[:length]
                if position != begin and overlap:
                    fade_len = min(overlap, length)
                    retained = max(1, int(fade_len * lgr_num))
                    left = (fade_len - retained) // 2
                    fade = np.concatenate([
                        np.zeros(left), np.linspace(0, 1, retained),
                        np.ones(fade_len - retained - left),
                    ])
                    converted[:fade_len] = (
                        output[position:position + fade_len] * (1 - fade)
                        + converted[:fade_len] * fade
                    )
                output[position:stop] = converted
                if stop == end:
                    break
                position += window - overlap
        return output

    def clear_empty(self):
        gc.collect()
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()

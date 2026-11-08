from pathlib import Path

import torch
from torch import nn

from vencoder.encoder import SpeechEncoder
from vencoder.whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
from vencoder.whisper.model import AudioEncoder, ModelDimensions


def convert_linear_to_int4(module):
    import bitsandbytes as bnb

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            layer = bnb.nn.Linear4bit(
                child.in_features, child.out_features, bias=child.bias is not None,
                compute_dtype=torch.float16, quant_type="nf4", compress_statistics=True,
            )
            layer.weight = bnb.nn.Params4bit(
                child.weight.detach(), requires_grad=False,
                quant_type="nf4", compress_statistics=True,
            )
            if child.bias is not None:
                layer.bias = nn.Parameter(child.bias.detach(), requires_grad=False)
            setattr(module, name, layer)
        else:
            convert_linear_to_int4(child)


class WhisperPPGLargeV3(SpeechEncoder):
    def __init__(self, vec_path="pretrain/large-v3.pt", device=None, precision=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if precision is None:
            if self.dev.type != "cuda":
                precision = "fp32"
            else:
                # Current bitsandbytes CUDA binaries do not include Pascal kernels.
                precision = "fp16" if torch.cuda.get_device_capability(self.dev)[0] < 7 else "int4"
        if precision not in {"int4", "fp16", "fp32"}:
            raise ValueError("Whisper precision must be int4, fp16 or fp32")
        if self.dev.type != "cuda" and precision != "fp32":
            raise ValueError("CPU encoding requires fp32 precision")
        if not Path(vec_path).is_file():
            raise FileNotFoundError(f"Missing Whisper weights: {vec_path}")
        checkpoint = torch.load(vec_path, map_location="cpu", weights_only=True, mmap=True)
        dims = ModelDimensions(**checkpoint["dims"])
        self.hidden_dim = dims.n_audio_state
        self.n_mels = dims.n_mels
        with torch.device("meta"):
            self.model = AudioEncoder(
                dims.n_mels, dims.n_audio_ctx, dims.n_audio_state,
                dims.n_audio_head, dims.n_audio_layer,
            )
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("encoder.")
        }
        self.model.load_state_dict(encoder_state, strict=True, assign=True)
        self.dtype = torch.float32 if precision == "fp32" else torch.float16
        self.model.to(dtype=self.dtype)
        if precision == "int4":
            convert_linear_to_int4(self.model)
        self.model.eval().requires_grad_(False).to(self.dev)

    @torch.inference_mode()
    def encoder(self, wav):
        if wav.ndim != 1 or not 0 < wav.numel() <= N_SAMPLES:
            raise ValueError("Whisper expects a mono 16 kHz waveform of at most 30 seconds")
        n_frames = wav.numel() // 320
        if n_frames == 0:
            raise ValueError("Audio must contain at least 320 samples")
        mel = log_mel_spectrogram(pad_or_trim(wav.float()), self.n_mels)
        ppg = self.model(mel.to(device=self.dev, dtype=self.dtype).unsqueeze(0))
        return ppg[:, :n_frames].transpose(1, 2).float()

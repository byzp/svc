import torch
import torch.nn.functional as F

from modules.F0Predictor.audio import resample_audio

from .constants import *  # noqa: F403
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_cents, to_viterbi_cents


class RMVPE:
    def __init__(self, model_path, device=None, dtype = torch.float32, hop_length=160):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        model = E2E0(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt['model'])
        model = model.to(dtype).to(self.device)
        model.eval()
        self.model = model
        self.dtype = dtype
        self.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)  # noqa: F405

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='constant')
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            cents_pred = to_viterbi_cents(hidden, thred=thred)
        else:
            cents_pred = to_local_average_cents(hidden, thred=thred)
        f0 = torch.where(cents_pred > 0, 10 * (2 ** (cents_pred / 1200)), 0)
        return f0

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.05, use_viterbi=False):
        # Keep the FFT frontend in FP32 even when network weights use FP16.
        audio = audio.unsqueeze(0).float().to(self.device)
        audio_res = resample_audio(audio, sample_rate, 16000)
        mel_extractor = self.mel_extractor.to(self.device)
        mel = mel_extractor(audio_res, center=True).to(self.dtype)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden.squeeze(0), thred=thred, use_viterbi=use_viterbi)
        return f0

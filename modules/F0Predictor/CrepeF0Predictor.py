import numpy as np
import torch

from modules.F0Predictor.crepe import CrepePitchExtractor
from modules.F0Predictor.F0Predictor import F0Predictor


class CrepeF0Predictor(F0Predictor):
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, device=None,
                 sampling_rate=44100, threshold=0.05, model="full"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.F0Creper = CrepePitchExtractor(
            hop_length=hop_length, f0_min=f0_min, f0_max=f0_max,
            device=self.device, threshold=threshold, model=model)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.name = "crepe"

    @torch.inference_mode()
    def compute_f0_uv(self, wav, p_len=None):
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1 or not np.isfinite(wav).all():
            raise ValueError("Expected finite mono audio")
        natural_len = wav.size // self.hop_length
        if p_len is None:
            p_len = natural_len
        elif p_len < 0 or abs(p_len - natural_len) >= 4:
            raise ValueError("pad length error")
        if not p_len or not wav.size or not np.any(wav):
            empty = np.zeros(p_len, dtype=np.float32)
            return empty, empty.copy()
        x = torch.as_tensor(wav, dtype=torch.float32, device=self.device)
        return self.F0Creper(x[None], self.sampling_rate, pad_to=p_len)

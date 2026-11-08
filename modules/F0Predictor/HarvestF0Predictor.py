import numpy as np
import pyworld

from modules.F0Predictor.F0Predictor import F0Predictor


class HarvestF0Predictor(F0Predictor):
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "harvest"

    def compute_f0_uv(self, wav, p_len=None):
        wav = np.array(wav, dtype=np.float64, order="C", copy=True)
        if wav.ndim != 1 or not np.isfinite(wav).all():
            raise ValueError("Expected finite mono audio")
        if p_len is None:
            p_len = wav.size // self.hop_length
        if p_len < 0:
            raise ValueError("p_len must be nonnegative")
        if not p_len or not wav.size or not np.any(wav):
            empty = np.zeros(p_len, dtype=np.float32)
            return empty, empty.copy()
        f0, t = pyworld.harvest(
            wav, fs=self.sampling_rate,
            f0_floor=self.f0_min, f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav, f0, t, self.sampling_rate)
        # Preserve unvoiced boundaries while aligning WORLD's extra last frame.
        f0[f0 < 0.001] = np.nan
        f0 = np.interp(np.arange(p_len) * len(f0) / p_len, np.arange(len(f0)), f0)
        return self.interpolate_f0(np.nan_to_num(f0))

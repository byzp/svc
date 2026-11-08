import numpy as np
import torch

from modules.F0Predictor.F0Predictor import F0Predictor

from .rmvpe import RMVPE


class RMVPEF0Predictor(F0Predictor):
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, dtype=torch.float32,
                 device=None, sampling_rate=44100, threshold=0.05):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rmvpe = RMVPE(model_path="pretrain/rmvpe.pt", dtype=dtype, device=self.device)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.name = "rmvpe"

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
        x = torch.as_tensor(wav, dtype=self.dtype, device=self.device)
        # Reflection padding in the spectral frontend needs at least one window.
        if x.numel() < self.sampling_rate // 10:
            x = torch.nn.functional.pad(x, (0, self.sampling_rate // 10 - x.numel()))
        f0 = self.rmvpe.infer_from_audio(x, self.sampling_rate, self.threshold)
        # Drop frames added solely to support very short input audio.
        if x.numel() > wav.size:
            frames = max(1, int(np.ceil(wav.size / self.sampling_rate * 100)))
            f0 = f0[:frames]
        return self.post_process_f0(f0.float().cpu().numpy(), p_len)

import numpy as np


class F0Predictor:
    def compute_f0(self, wav, p_len=None):
        return self.compute_f0_uv(wav, p_len)[0]

    def compute_f0_uv(self, wav, p_len=None):
        raise NotImplementedError

    @staticmethod
    def interpolate_f0(f0):
        """Fill unvoiced F0 values while preserving the original voiced mask."""
        f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
        voiced = np.isfinite(f0) & (f0 > 0)
        uv = voiced.astype(np.float32)
        if not voiced.any():
            return np.zeros_like(f0), uv
        indices = np.flatnonzero(voiced)
        return np.interp(np.arange(f0.size), indices, f0[indices]).astype(np.float32), uv

    def post_process_f0(self, f0, p_len):
        """Align neural predictor frames to the requested SVC frame count."""
        f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
        if p_len < 0:
            raise ValueError("p_len must be nonnegative")
        if not f0.size or not p_len:
            empty = np.zeros(p_len, dtype=np.float32)
            return empty, empty.copy()
        # Match torch.interpolate(mode='nearest'), including unvoiced frames.
        indices = np.minimum(np.arange(p_len) * f0.size // p_len, f0.size - 1)
        return self.interpolate_f0(f0[indices])

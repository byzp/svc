from math import gcd

import numpy as np
import torch
from scipy.signal import resample_poly


def resample_audio(audio, orig_sr, target_sr):
    """Resample F0 input without a compiled torchaudio/PyTorch ABI dependency."""
    if orig_sr == target_sr:
        return audio
    factor = gcd(orig_sr, target_sr)
    samples = audio.detach().float().cpu().numpy()
    samples = resample_poly(samples, target_sr // factor, orig_sr // factor, axis=-1,
                            window=("kaiser", 14.0))
    return torch.from_numpy(np.ascontiguousarray(samples)).to(device=audio.device, dtype=audio.dtype)

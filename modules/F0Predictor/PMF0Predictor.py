import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from modules.F0Predictor.F0Predictor import F0Predictor


class PMF0Predictor(F0Predictor):
    """Window-corrected autocorrelation pitch tracker, without Praat bindings.

    Keeps the historical pm selector and frame/F0/UV interface. This is a
    NumPy/SciPy implementation, not a bit-exact reproduction of Praat.
    """

    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, sampling_rate=44100):
        if hop_length <= 0 or not 0 < f0_min < f0_max < sampling_rate / 2:
            raise ValueError("Invalid hop length or pitch range")
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "pm"
        self.voicing_threshold = 0.6
        # Three periods at the pitch floor, as in Praat's autocorrelation mode.
        self.frame_length = int(np.ceil(3 * sampling_rate / f0_min))
        self.frame_length += self.frame_length % 2
        self.window = np.hanning(self.frame_length)
        self.fft_length = next_fast_len(2 * self.frame_length - 1)
        spectrum = rfft(self.window, n=self.fft_length)
        self.window_ac = irfft(spectrum * spectrum.conj(), n=self.fft_length)[:self.frame_length]
        self.window_ac /= self.window_ac[0]

    def compute_f0_uv(self, wav, p_len=None):
        wav = np.asarray(wav, dtype=np.float64)
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

        half = self.frame_length // 2
        padded = np.pad(wav, (half, half + max(0, p_len - natural_len) * self.hop_length))
        frames = np.lib.stride_tricks.sliding_window_view(padded, self.frame_length)[::self.hop_length][:p_len]
        min_lag = max(1, int(np.floor(self.sampling_rate / self.f0_max)))
        max_lag = int(np.ceil(self.sampling_rate / self.f0_min))
        # Each frame has up to 15 voiced candidates and one unvoiced candidate.
        frequencies = np.zeros((p_len, 16))
        scores = np.full((p_len, 16), -np.inf)
        scores[:, 0] = self.voicing_threshold
        peak = np.max(np.abs(wav))
        for start in range(0, p_len, 128):
            block = frames[start:start + 128].copy()
            block -= block.mean(axis=1, keepdims=True)
            rms = np.sqrt(np.mean(block ** 2, axis=1))
            spectrum = rfft(block * self.window, n=self.fft_length, axis=1)
            ac = irfft(spectrum * spectrum.conj(), n=self.fft_length, axis=1)[:, :max_lag + 2]
            energy = ac[:, :1].copy()
            ac /= np.maximum(energy, np.finfo(float).tiny)
            ac /= self.window_ac[:max_lag + 2]
            for offset, row in enumerate(ac):
                if rms[offset] < 0.03 * peak:
                    continue
                lags = np.arange(min_lag, max_lag + 1)
                lags = lags[(row[lags] > row[lags - 1]) & (row[lags] >= row[lags + 1])]
                if not lags.size:
                    continue
                # Parabolic refinement gives sub-sample pitch estimates.
                left, middle, right = row[lags - 1], row[lags], row[lags + 1]
                delta = np.clip(0.5 * (left - right) / (left - 2 * middle + right), -0.5, 0.5)
                hz = self.sampling_rate / (lags + delta)
                strength = np.minimum(1, middle - 0.25 * (left - right) * delta)
                strength -= 0.01 * np.log2(self.f0_max / hz)
                # Allow interpolation roundoff at the configured pitch limits.
                valid = (hz >= self.f0_min * 0.999) & (hz <= self.f0_max * 1.001) & (strength > 0.45)
                hz, strength = np.clip(hz[valid], self.f0_min, self.f0_max), strength[valid]
                best = np.argsort(strength)[-15:]
                index = start + offset
                frequencies[index, 1:1 + best.size] = hz[best]
                scores[index, 1:1 + best.size] = strength[best]

        # Penalize octave jumps and voiced/unvoiced transitions between frames.
        back = np.zeros_like(frequencies, dtype=np.int32)
        previous = scores[0].copy()
        transition_scale = 0.01 / (self.hop_length / self.sampling_rate)
        for index in range(1, p_len):
            old = frequencies[index - 1, :, None]
            new = frequencies[index, None, :]
            both_voiced = (old > 0) & (new > 0)
            jumps = np.abs(np.log2(np.maximum(old, 1) / np.maximum(new, 1)))
            cost = np.where(both_voiced, 0.35 * jumps, 0.14 * ((old > 0) != (new > 0)))
            paths = previous[:, None] - cost * transition_scale
            back[index] = np.argmax(paths, axis=0)
            previous = scores[index] + np.max(paths, axis=0)
        f0 = np.zeros(p_len, dtype=np.float32)
        state = int(np.argmax(previous))
        for index in range(p_len - 1, -1, -1):
            f0[index] = frequencies[index, state]
            state = back[index, state]
        return self.interpolate_f0(f0)

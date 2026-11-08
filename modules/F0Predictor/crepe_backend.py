"""CREPE inference adapted from torchcrepe (MIT; see CREPE_LICENSE).

Use the model assets bundled with torchcrepe without importing its torchaudio
file loader. Audio is already decoded by the preprocessing/inference entrypoint.
"""
from importlib.metadata import distribution

import librosa
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .audio import resample_audio


class Crepe(nn.Module):
    def __init__(self, capacity="full"):
        super().__init__()
        if capacity == "full":
            channels, features = [1, 1024, 128, 128, 128, 256, 512], 2048
        elif capacity == "tiny":
            channels, features = [1, 128, 16, 16, 16, 32, 64], 256
        else:
            raise ValueError(f"Unknown CREPE capacity: {capacity}")
        self.features = features
        for index in range(6):
            kernel, stride = (512, 4) if index == 0 else (64, 1)
            setattr(self, f"conv{index + 1}", nn.Conv2d(
                channels[index], channels[index + 1], (kernel, 1), (stride, 1)))
            setattr(self, f"conv{index + 1}_BN", nn.BatchNorm2d(
                channels[index + 1], eps=0.0010000000474974513, momentum=0.0))
        self.classifier = nn.Linear(features, 360)

    def forward(self, frames):
        x = frames[:, None, :, None]
        for index in range(1, 7):
            padding = (0, 0, 254, 254) if index == 1 else (0, 0, 31, 32)
            x = F.relu(getattr(self, f"conv{index}")(F.pad(x, padding)))
            x = getattr(self, f"conv{index}_BN")(x)
            x = F.max_pool2d(x, (2, 1), (2, 1))
        x = x.permute(0, 2, 1, 3).reshape(-1, self.features)
        return torch.sigmoid(self.classifier(x))


class CrepeInference:
    def __init__(self, capacity, device, decoder="viterbi"):
        if decoder not in {"viterbi", "argmax", "weighted_argmax"}:
            raise ValueError(f"Unknown CREPE decoder: {decoder}")
        self.device = torch.device(device)
        self.decoder = decoder
        self.model = Crepe(capacity)
        weights = distribution("torchcrepe").locate_file(f"torchcrepe/assets/{capacity}.pth")
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.eval().requires_grad_(False).to(self.device)
        bins = np.arange(360)
        transition = np.maximum(12 - np.abs(bins[:, None] - bins[None, :]), 0)
        self.transition = transition / transition.sum(axis=1, keepdims=True)

    @torch.inference_mode()
    def __call__(self, audio, sampling_rate, hop_length, fmin, fmax):
        audio = resample_audio(audio.float(), sampling_rate, 16000)
        hop = max(1, int(hop_length * 16000 / sampling_rate))
        frames = F.pad(audio, (512, 512)).unfold(-1, 1024, hop)[0]
        outputs = []
        for block in frames.split(128):
            block = block.to(self.device).clone()
            block -= block.mean(dim=1, keepdim=True)
            block /= block.std(dim=1, keepdim=True).clamp_min(1e-10)
            outputs.append(self.model(block))
        probabilities = torch.cat(outputs).T
        low = max(0, int(np.floor((1200 * np.log2(fmin / 10) - 1997.3794084376191) / 20)))
        high = min(360, int(np.ceil((1200 * np.log2(fmax / 10) - 1997.3794084376191) / 20)))
        probabilities[:low] = -float("inf")
        probabilities[high:] = -float("inf")
        if self.decoder == "viterbi":
            path = librosa.sequence.viterbi(
                probabilities.softmax(dim=0).cpu().numpy(), self.transition)
            bins = torch.as_tensor(path, device=self.device, dtype=torch.long)
        else:
            bins = probabilities.argmax(dim=0)
        periodicity = probabilities.gather(0, bins[None])[0]
        if self.decoder == "weighted_argmax":
            positions = torch.arange(360, device=self.device)[:, None]
            weights = torch.sigmoid(probabilities) * ((positions - bins[None]).abs() <= 4)
            cents = (weights * (20 * positions + 1997.3794084376191)).sum(0) / weights.sum(0)
        else:
            # Same triangular quantization dither as torchcrepe.
            cents = 20 * bins + 1997.3794084376191
            cents = cents + cents.new_tensor(np.random.triangular(-20, 0, 20, size=bins.shape))
        pitch = 10 * 2 ** (cents / 1200)
        return pitch[None], periodicity[None]


def silence_mask(audio, sampling_rate, hop_length):
    """The A-weighted -60 dB silence mask used by torchcrepe."""
    audio = resample_audio(audio.float(), sampling_rate, 16000)
    hop = max(1, int(hop_length * 16000 / sampling_rate))
    samples = np.pad(audio[0].cpu().numpy(), (512, 512))
    spectrum = librosa.stft(samples, n_fft=1024, hop_length=hop,
                           win_length=1024, center=False)
    frequencies = librosa.fft_frequencies(sr=16000, n_fft=1024)
    # The DC A-weight is -inf by definition, then clamped by the loudness floor.
    with np.errstate(divide="ignore"):
        weights = librosa.A_weighting(frequencies)[:, None] - 20
    loudness = np.maximum(librosa.amplitude_to_db(np.abs(spectrum)) + weights, -100).mean(axis=0)
    return torch.as_tensor(loudness < -60, device=audio.device)[None]

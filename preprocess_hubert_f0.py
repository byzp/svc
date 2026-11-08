import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

import utils
from modules.F0Predictor import F0_PREDICTORS

logger = logging.getLogger(__name__)


def process_one(filename, hmodel, f0_predictor, device, hps, volume_extractor):
    import librosa
    from modules.mel_processing import spectrogram_torch

    wav, _ = librosa.load(filename, sr=hps.data.sampling_rate, mono=True)
    audio_norm = torch.from_numpy(wav)[None]
    soft_path = filename + ".soft.pt"
    if not Path(soft_path).exists():
        wav16k = librosa.resample(wav, orig_sr=hps.data.sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        # Encode the whole file without exceeding Whisper's 30-second context.
        features = [hmodel.encoder(chunk).cpu() for chunk in wav16k.split(480000) if chunk.numel() >= 320]
        if not features:
            raise ValueError(f"Audio is too short: {filename}")
        torch.save(torch.cat(features, dim=-1), soft_path)
    f0_path = filename + ".f0.npy"
    if not Path(f0_path).exists():
        f0, uv = f0_predictor.compute_f0_uv(wav)
        np.save(f0_path, np.stack([f0, uv]))
    spec_path = str(Path(filename).with_suffix(".spec.pt"))
    if not Path(spec_path).exists():
        spec = spectrogram_torch(
            audio_norm, hps.data.filter_length, hps.data.sampling_rate,
            hps.data.hop_length, hps.data.win_length, center=False,
        )
        torch.save(spec.squeeze(0), spec_path)
    if hps.model.vol_embedding:
        volume_path = filename + ".vol.npy"
        if not Path(volume_path).exists():
            np.save(volume_path, volume_extractor.extract(audio_norm).cpu().numpy())


def worker_process(rank, chunks, args):
    hps = utils.get_hparams_from_file(args.config)
    if args.force_cpu or args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    hmodel = utils.get_speech_encoder(
        hps.model.speech_encoder, device=device, precision=args.encoder_precision,
    )
    predictor = utils.get_f0_predictor(
        args.f0_predictor, hop_length=hps.data.hop_length, sampling_rate=hps.data.sampling_rate,
        device=device, threshold=args.f0_filter_threshold,
    )
    volume_extractor = utils.Volume_Extractor(hps.data.hop_length)
    for filename in tqdm(chunks[rank], desc=f"Features ({device})", position=rank):
        process_one(filename, hmodel, predictor, device, hps, volume_extractor)


def main():
    parser = argparse.ArgumentParser(description="Extract Whisper large-v3, F0, spectrogram and volume features")
    parser.add_argument("-c", "--config", default="configs/config.json")
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--in_dir", default="dataset/44k")
    parser.add_argument("--f0_predictor", choices=F0_PREDICTORS, default="pm")
    parser.add_argument("--f0_filter_threshold", type=float, default=0.05,
                        help="Voicing threshold for crepe, rmvpe and fcpe (0 to 1)")
    parser.add_argument("--encoder_precision", choices=["fp32", "fp16", "int4"], default=None)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()
    if args.num_processes < 1:
        parser.error("--num_processes must be positive")
    if not 0 <= args.f0_filter_threshold <= 1:
        parser.error("--f0_filter_threshold must be between 0 and 1")
    filenames = sorted(str(path) for path in Path(args.in_dir).glob("*/*.wav"))
    if not filenames:
        parser.error(f"No WAV files found in {args.in_dir}")
    n_workers = min(args.num_processes, len(filenames))
    chunks = [filenames[index::n_workers] for index in range(n_workers)]
    if n_workers == 1:
        worker_process(0, chunks, args)
    else:
        mp.spawn(worker_process, args=(chunks, args), nprocs=n_workers, join=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

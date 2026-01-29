import argparse
import logging
import os
import random
from glob import glob
from random import shuffle

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

import diffusion.logger.utils as du
import utils
from diffusion.vocoder import Vocoder
from modules.mel_processing import spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

hps = utils.get_hparams_from_file("configs/config.json")
dconfig = du.load_config("configs/diffusion.yaml")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps["model"]["speech_encoder"]


def process_one(filename, hmodel, f0_predictor, device, diff, mel_extractor, volume_extractor):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav).unsqueeze(0)
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        with torch.no_grad():
            c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)
    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0, uv = f0_predictor.compute_f0_uv(wav)
        np.save(f0_path, np.asanyarray((f0, uv), dtype=object))
    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        if sr != hps.data.sampling_rate:
            raise ValueError(f"{sr} SR doesn't match target {hps.data.sampling_rate} SR")
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_path)
    if diff or hps.model.vol_embedding:
        volume_path = filename + ".vol.npy"
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            np.save(volume_path, volume.cpu().numpy())
    if diff and mel_extractor is not None:
        mel_path = filename + ".mel.npy"
        if not os.path.exists(mel_path):
            with torch.no_grad():
                mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
            mel = mel_t.squeeze().cpu().numpy()
            np.save(mel_path, mel)
        aug_mel_path = filename + ".aug_mel.npy"
        aug_vol_path = filename + ".aug_vol.npy"
        if not os.path.exists(aug_mel_path) or not os.path.exists(aug_vol_path):
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            keyshift = random.uniform(-5, 5)
            aug_audio = audio_norm * (10 ** log10_vol_shift)
            with torch.no_grad():
                aug_mel_t = mel_extractor.extract(aug_audio.to(device), sampling_rate, keyshift=keyshift)
            aug_mel = aug_mel_t.squeeze().cpu().numpy()
            aug_vol = volume_extractor.extract(aug_audio)
            if not os.path.exists(aug_mel_path):
                np.save(aug_mel_path, np.asanyarray((aug_mel, keyshift), dtype=object))
            if not os.path.exists(aug_vol_path):
                np.save(aug_vol_path, aug_vol.cpu().numpy())


def worker_process(rank, file_chunk, num_gpus, f0p, diff, mel_extractor_cfg, progress_queue, gpu_id):
    try:
        if torch.cuda.is_available():
            gpu_label = gpu_id if (num_gpus > 0 and gpu_id is not None) else "N/A"
            device = torch.device(f"cuda:{gpu_label}")
            logger.info(f"Worker {rank} initialized on {device} (GPU {gpu_label})")
        else:
            device = torch.device("cpu")
        hmodel = utils.get_speech_encoder(speech_encoder, device=device)
        f0_device = device if f0p in ['rmvpe', 'fcpe', 'crepe'] else None
        f0_predictor = utils.get_f0_predictor(
            f0p,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            device=f0_device,
            threshold=0.05
        )
        volume_extractor = utils.Volume_Extractor(hop_length)
        mel_extractor = None
        if diff and mel_extractor_cfg is not None:
            try:
                mel_extractor = Vocoder(mel_extractor_cfg[0], mel_extractor_cfg[1], device=device)
            except Exception as e:
                logger.error(f"Worker {rank} failed to load mel extractor: {e}")
        logger.info(f"Worker {rank} models loaded successfully")
        for i, filename in enumerate(file_chunk):
            try:
                process_one(filename, hmodel, f0_predictor, device, diff, mel_extractor, volume_extractor)
                if device.type == "cuda" and (i + 1) % 50 == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Worker {rank} error processing {filename}: {e}")
            progress_queue.put(1)
        del hmodel, f0_predictor, mel_extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Worker {rank} fatal error: {e}")
        import traceback
        traceback.print_exc()


def progress_listener(progress_queue, total_files):
    pbar = tqdm(total=total_files, desc="Processing", position=0)
    completed = 0
    while completed < total_files:
        try:
            progress_queue.get(timeout=60)
            completed += 1
            pbar.update(1)
        except:
            break
    pbar.close()


def chunkify(seq, k):
    if k <= 0:
        return [seq]
    n = len(seq)
    q, r = divmod(n, k)
    chunks = []
    start = 0
    for i in range(k):
        sz = q + (1 if i < r else 0)
        chunks.append(seq[start:start + sz])
        start += sz
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument("--in_dir", type=str, default="dataset/44k", help="path to input dir")
    parser.add_argument('--use_diff', action='store_true', help='Use diffusion model')
    parser.add_argument('--f0_predictor', type=str, default="rmvpe",
                        help='F0 predictor: crepe, pm, dio, harvest, rmvpe, fcpe')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU processing')
    args = parser.parse_args()

    logger.info(f"SpeechEncoder: {speech_encoder}")
    logger.info(f"F0 Predictor: {args.f0_predictor}")
    logger.info(f"Diffusion Mode: {args.use_diff}")

    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)
    if not filenames:
        logger.error(f"No wav files found in {args.in_dir}")
        return
    shuffle(filenames)
    total_files = len(filenames)
    logger.info(f"Found {total_files} files")

    if args.force_cpu:
        num_gpus = 0
    else:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    num_processes = args.num_processes
    if num_processes <= 0:
        num_processes = max(1, min(os.cpu_count() or 1, num_gpus if num_gpus > 0 else 4))
    if num_gpus > 0:
        num_processes = min(num_processes, num_gpus * 2)
    num_processes = min(num_processes, total_files)

    logger.info(f"Using {num_processes} processes, {num_gpus} GPUs")

    mel_extractor_cfg = None
    if args.use_diff:
        mel_extractor_cfg = (dconfig.vocoder.type, dconfig.vocoder.ckpt)

    mp.set_start_method("spawn", force=True)

    if num_gpus > 0:
        gpu_ranks_map = {g: [] for g in range(num_gpus)}
        for rank in range(num_processes):
            gpu_id = rank % num_gpus
            gpu_ranks_map[gpu_id].append(rank)
        gpu_files = {g: [] for g in range(num_gpus)}
        for i, fn in enumerate(filenames):
            gpu_files[i % num_gpus].append(fn)
        rank_chunks = [[] for _ in range(num_processes)]
        for g in range(num_gpus):
            ranks_for_gpu = gpu_ranks_map[g]
            if not ranks_for_gpu:
                continue
            chunks_for_gpu = chunkify(gpu_files[g], len(ranks_for_gpu))
            for idx, r in enumerate(ranks_for_gpu):
                rank_chunks[r] = chunks_for_gpu[idx] if idx < len(chunks_for_gpu) else []
    else:
        rank_chunks = chunkify(filenames, num_processes)

    progress_queue = mp.Queue()
    processes = []
    for rank in range(num_processes):
        if num_gpus > 0:
            gpu_id = rank % num_gpus
        else:
            gpu_id = None
        p = mp.Process(
            target=worker_process,
            args=(rank, rank_chunks[rank], num_gpus, args.f0_predictor,
                  args.use_diff, mel_extractor_cfg, progress_queue, gpu_id)
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker {rank} with {len(rank_chunks[rank])} files (GPU {gpu_id if gpu_id is not None else 'N/A'})")

    progress_proc = mp.Process(target=progress_listener, args=(progress_queue, total_files))
    progress_proc.start()

    for p in processes:
        p.join()

    progress_queue.put(None)
    progress_proc.join(timeout=5)
    if progress_proc.is_alive():
        progress_proc.terminate()


if __name__ == "__main__":
    main()

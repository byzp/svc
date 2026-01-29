import argparse
import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import librosa
import numpy as np
from rich.progress import track
import soundfile as sf


def load_wav(wav_path):
    # keep default mono=True; if you want to preserve channels use mono=False
    return librosa.load(wav_path, sr=None)


def trim_wav(wav, top_db=40):
    return librosa.effects.trim(wav, top_db=top_db)


def normalize_peak(wav, threshold=1.0):
    peak = np.abs(wav).max()
    if peak > threshold and peak > 0:
        wav = 0.98 * wav / peak
    return wav


def resample_wav(wav, sr, target_sr):
    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


def save_wav_to_path(wav, save_path, sr):
    """
    Use soundfile.write so that output format is derived from filename extension.
    soundfile accepts float arrays in range [-1, 1]. If the user wants specific
    PCM subtype they can change the call (e.g. subtype='PCM_16').
    """
    # ensure float32 for soundfile
    wav_to_write = wav.astype(np.float32)
    # soundfile will infer format from extension ('.wav', '.flac', etc.)
    sf.write(save_path, wav_to_write, sr)


def process(item):
    spkdir, wav_name, args = item
    speaker = spkdir.replace("\\", "/").split("/")[-1]

    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    # only accept .wav or .flac (case-insensitive)
    if os.path.exists(wav_path) and wav_path.lower().endswith(('.wav', '.flac')):
        os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)

        wav, sr = load_wav(wav_path)
        wav, _ = trim_wav(wav)
        wav = normalize_peak(wav)
        resampled_wav = resample_wav(wav, sr, args.sr2)

        if not args.skip_loudnorm:
            denom = np.max(np.abs(resampled_wav))
            if denom > 0:
                resampled_wav = resampled_wav / denom

        # preserve original extension for output (so .flac -> .flac, .wav -> .wav)
        save_path2 = os.path.join(args.out_dir2, speaker, wav_name)
        save_wav_to_path(resampled_wav, save_path2, args.sr2)


def process_all_speakers():
    process_count = 30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for speaker in speakers:
            spk_dir = os.path.join(args.in_dir, speaker)
            if os.path.isdir(spk_dir):
                print(spk_dir)
                # list only wav/flac files (case-insensitive)
                files = [f for f in os.listdir(spk_dir) if f.lower().endswith(('.wav', '.flac'))]
                futures = [executor.submit(process, (spk_dir, i, args)) for i in files]
                for _ in track(concurrent.futures.as_completed(futures), total=len(futures), description="resampling:"):
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr2", type=int, default=44100, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="./dataset_raw", help="path to source dir")
    parser.add_argument("--out_dir2", type=str, default="./dataset/44k", help="path to target dir")
    parser.add_argument("--skip_loudnorm", action="store_true", help="Skip loudness matching if you have done it")
    args = parser.parse_args()

    print(f"CPU count: {cpu_count()}")
    speakers = os.listdir(args.in_dir)
    process_all_speakers()

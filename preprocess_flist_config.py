import argparse
import json
import random
import wave
from pathlib import Path


def get_wav_duration(file_path):
    with wave.open(str(file_path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def main():
    parser = argparse.ArgumentParser(description="Create filelists using the existing Whisper large-v3 model configuration")
    parser.add_argument("-c", "--config", default="configs/config.json")
    parser.add_argument("--train_list", default=None)
    parser.add_argument("--val_list", default=None)
    parser.add_argument("--source_dir", default="dataset/44k")
    parser.add_argument("--speech_encoder", choices=["whisper-ppg-large-v3"], default="whisper-ppg-large-v3")
    parser.add_argument("--vol_aug", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    model = config["model"]
    if model["speech_encoder"] != args.speech_encoder or model["ssl_dim"] != 1280 or model["vocoder_name"] != "nsf-hifigan":
        parser.error("Configuration must match the Whisper large-v3 model in logs/44k")
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        parser.error(f"Missing dataset directory: {source_dir}")
    rng = random.Random(config["train"]["seed"])
    train, val, speakers = [], [], {}
    for speaker in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        wavs = [str(path) for path in sorted(speaker.glob("*.wav")) if get_wav_duration(path) >= 0.3]
        if len(wavs) < 2:
            parser.error(f"Speaker {speaker.name} needs at least two WAV files of 0.3 seconds or longer")
        speakers[speaker.name] = len(speakers)
        rng.shuffle(wavs)
        val_count = min(2, len(wavs) - 1)
        val.extend(wavs[:val_count])
        train.extend(wavs[val_count:])
    if not train or not val:
        parser.error("Training and validation filelists must both contain audio")
    rng.shuffle(train)
    rng.shuffle(val)
    train_path = Path(args.train_list or config["data"]["training_files"])
    val_path = Path(args.val_list or config["data"]["validation_files"])
    if train_path.resolve() == val_path.resolve():
        parser.error("Training and validation filelists must have different paths")
    for path, values in [(train_path, train), (val_path, val)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(values) + "\n")
    config["data"]["training_files"] = str(train_path)
    config["data"]["validation_files"] = str(val_path)
    config["spk"] = speakers
    model["n_speakers"] = len(speakers)
    if args.vol_aug:
        config["train"]["vol_aug"] = True
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"Training: {len(train)}, validation: {len(val)}, speakers: {speakers}")
    print(f"Configuration updated: {config_path}")


if __name__ == "__main__":
    main()

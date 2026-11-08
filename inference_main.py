import argparse
from pathlib import Path

from modules.F0Predictor import F0_PREDICTORS


def main():
    parser = argparse.ArgumentParser(description="Whisper large-v3 So-VITS-SVC inference")
    parser.add_argument("-m", "--model_path", default="logs/44k/G_0.pth")
    parser.add_argument("-c", "--config_path", default="configs/config.json")
    parser.add_argument("-n", "--clean_names", nargs="+", required=True, help="Input paths or filenames in raw/")
    parser.add_argument("-s", "--spk_list", nargs="+", help="Speaker names; defaults to the first configured speaker")
    parser.add_argument("-t", "--trans", nargs="+", type=int, default=[0], help="Pitch shift in semitones")
    parser.add_argument("-d", "--device", default=None, help="cpu or cuda; selected automatically by default")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--encoder_precision", choices=["fp32", "fp16", "int4"], default=None)
    parser.add_argument("-f0p", "--f0_predictor", choices=F0_PREDICTORS, default="rmvpe")
    parser.add_argument("-ft", "--f0_filter_threshold", type=float, default=0.05,
                        help="Voicing threshold for crepe, rmvpe and fcpe (0 to 1)")
    parser.add_argument("-a", "--auto_predict_f0", action="store_true")
    parser.add_argument("-eh", "--enhance", action="store_true", help="Enable NSF-HiFiGAN enhancement")
    parser.add_argument("-eak", "--enhancer_adaptive_key", type=int, default=0,
                        help="Adapt the enhancer to higher pitches (semitones)")
    parser.add_argument("-cl", "--clip", type=float, default=0, help="Clip length in seconds; 0 selects automatically")
    parser.add_argument("-sd", "--slice_db", type=float, default=-40)
    parser.add_argument("-ns", "--noice_scale", "--noise_scale", type=float, default=0.4)
    parser.add_argument("-p", "--pad_seconds", type=float, default=0.5)
    parser.add_argument("-lg", "--linear_gradient", type=float, default=0)
    parser.add_argument("-lgr", "--linear_gradient_retain", type=float, default=0.75)
    parser.add_argument("-lea", "--loudness_envelope_adjustment", type=float, default=1)
    parser.add_argument("-wf", "--wav_format", choices=["wav", "flac"], default="flac")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    if len(args.trans) not in {1, len(args.clean_names)}:
        parser.error("Provide one pitch shift or one per input file")
    if not 0 <= args.f0_filter_threshold <= 1:
        parser.error("--f0_filter_threshold must be between 0 and 1")

    import soundfile
    from inference.infer_tool import Svc

    svc = Svc(args.model_path, args.config_path, args.device, args.precision, args.encoder_precision,
              nsf_hifigan_enhance=args.enhance)
    speakers = args.spk_list or [next(iter(svc.spk2id.keys()))]
    shifts = args.trans * len(args.clean_names) if len(args.trans) == 1 else args.trans
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, shift in zip(args.clean_names, shifts):
        source = Path(name)
        if not source.is_file():
            source = Path("raw") / name
        if not source.suffix:
            source = source.with_suffix(".wav")
        if not source.is_file():
            raise FileNotFoundError(source)
        for speaker in speakers:
            audio = svc.slice_inference(
                source, speaker, shift, slice_db=args.slice_db,
                auto_predict_f0=args.auto_predict_f0, noice_scale=args.noice_scale,
                pad_seconds=args.pad_seconds, clip_seconds=args.clip,
                lg_num=args.linear_gradient, lgr_num=args.linear_gradient_retain,
                f0_predictor=args.f0_predictor,
                f0_filter_threshold=args.f0_filter_threshold,
                enhancer_adaptive_key=args.enhancer_adaptive_key,
                loudness_envelope_adjustment=args.loudness_envelope_adjustment,
            )
            target = output_dir / f"{source.stem}_{shift}key_{speaker}_{args.f0_predictor}.{args.wav_format}"
            soundfile.write(target, audio, svc.target_sample)
            print(target)
            svc.clear_empty()


if __name__ == "__main__":
    main()

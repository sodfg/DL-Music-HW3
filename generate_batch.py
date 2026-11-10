"""Generate one clip per test caption with the sample-code sampling loop (our cfg), saved as <id>.mp3.

    python generate_batch.py --out_dir generated_samples                 # baseline: scale 7, 100 steps, 10 s
    python generate_batch.py --out_dir outputs/scale_3 --scale 3         # guidance-scale ablation
    python generate_batch.py --out_dir outputs/apg_7 --guidance apg      # APG (eta 0, momentum -0.5)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from diffusers import StableAudioPipeline

sys.path.insert(0, str(Path(__file__).parent / "sample_code"))
from generate import generate  # noqa: E402  (uses cfg.py from sample_code)

from apg import generate_apg  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", default="sample_code/test.csv")
    parser.add_argument("--out_dir", default="generated_samples")
    parser.add_argument("--scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", choices=["cfg", "apg"], default="cfg")
    parser.add_argument("--eta", type=float, default=0.0, help="APG: weight of the parallel component")
    parser.add_argument("--momentum", type=float, default=-0.5, help="APG: momentum (0 = off)")
    parser.add_argument("--norm_threshold", type=float, default=0.0, help="APG: norm cap (0 = off)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(args.captions, newline="") as f:
        rows = list(csv.DictReader(f))

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    sr = pipe.vae.config.sampling_rate

    start = time.time()
    for n, row in enumerate(rows, 1):
        path = out_dir / f"{row['id']}.mp3"
        if path.exists():  # resumable: skip clips that are already done
            continue
        # Same seed for every caption, so runs with different settings start from the same noise.
        if args.guidance == "apg":
            audio = generate_apg(pipe, row["caption"], args.scale, args.steps, args.seconds, args.seed,
                                 eta=args.eta, momentum=args.momentum, norm_threshold=args.norm_threshold)
        else:
            audio = generate(pipe, row["caption"], args.scale, args.steps, args.seconds, args.seed)
        sf.write(path, audio.T.float().cpu().numpy(), sr, format="MP3")
        print(f"[{n}/{len(rows)}] {path.name}  ({time.time() - start:.0f}s elapsed)", flush=True)

    print(f"Done: {len(rows)} clips in {out_dir}")


if __name__ == "__main__":
    main()

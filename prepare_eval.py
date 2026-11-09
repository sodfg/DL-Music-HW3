"""Lay out audio and captions the way av-benchmark expects (wav files, paired by file name).

    python prepare_eval.py --pred generated_samples --exp baseline

Creates under --eval_dir:
    gt_wav/<id>.wav          the MusicCaps clip retrieved for caption <id> (reference for FAD / KL)
    text.csv                 name,caption (for the CLAP text-audio score)
    <exp>/pred_wav/<id>.wav  the generated clip for caption <id>
"""

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf


def mp3_to_wav(src, dst):
    if dst.exists():
        return
    audio, sr = sf.read(src)
    sf.write(dst, audio, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="directory of generated <id>.mp3")
    parser.add_argument("--exp", required=True, help="experiment name, e.g. baseline or scale_3")
    parser.add_argument("--eval_dir", default="outputs/eval")
    parser.add_argument("--retrieval", default="retrieval.json")
    parser.add_argument("--captions", default="sample_code/test.csv")
    parser.add_argument("--audio_dir", default="musiccaps/audio_mp3")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    with open(args.retrieval) as f:
        retrieval = json.load(f)
    with open(args.captions, newline="") as f:
        rows = list(csv.DictReader(f))

    # Shared by all experiments: references and captions
    gt_dir = eval_dir / "gt_wav"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        mp3_to_wav(Path(args.audio_dir) / f"{retrieval[row['id']]}.mp3", gt_dir / f"{row['id']}.wav")
    with open(eval_dir / "text.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "caption"])
        w.writerows([row["id"], row["caption"]] for row in rows)

    pred_dir = eval_dir / args.exp / "pred_wav"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        mp3_to_wav(Path(args.pred) / f"{row['id']}.mp3", pred_dir / f"{row['id']}.wav")

    print(f"{len(rows)} pairs ready: {gt_dir} <-> {pred_dir}")


if __name__ == "__main__":
    main()

"""Loudness / clipping statistics of generated clips, to quantify CFG oversaturation (results/clip_stats.csv).

The decoded mp3 can hold |x| > 1; those samples clip on playback and when converted to 16-bit wav for evaluation,
so we count clips whose peak exceeds 1.0 and the share of samples above it.

    python clip_stats.py generated_samples:cfg_s7 outputs/scale_15:cfg_s15 outputs/apg_s7:apg_s7 ...
"""

import csv
import sys

import numpy as np
import soundfile as sf


def main():
    rows = []
    for spec in sys.argv[1:]:
        folder, name = spec.split(":")
        peaks, rms, clipped_frac = [], [], []
        for i in range(100):
            audio, _ = sf.read(f"{folder}/{i}.mp3")
            peaks.append(np.abs(audio).max())
            rms.append(np.sqrt((audio**2).mean()))
            clipped_frac.append((np.abs(audio) > 1.0).mean())
        rows.append({
            "exp": name,
            "clips_with_peak>1": int(sum(p > 1.0 for p in peaks)),
            "mean_peak": float(np.mean(peaks)),
            "mean_rms": float(np.mean(rms)),
            "samples_over_1_%": float(100 * np.mean(clipped_frac)),
        })
        print(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()))

    with open("results/clip_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

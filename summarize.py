"""Collect results/metrics_*.json into one table (results/summary.csv) and print it.

    python summarize.py
"""

import csv
import json
from pathlib import Path

COLUMNS = ["FD-VGG", "KL-PASST-softmax", "LAION-CLAP-Score", "FD-PANN", "FD-PASST", "KL-PANNS-softmax",
           "MS-CLAP-Score", "ISC-PASST-mean"]


def main():
    rows = []
    for path in sorted(Path("results").glob("metrics_*.json")):
        metrics = json.loads(path.read_text())
        rows.append({"exp": path.stem.removeprefix("metrics_"), **{k: metrics.get(k) for k in COLUMNS}})

    with open("results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exp", *COLUMNS])
        w.writeheader()
        w.writerows(rows)

    print(f"{'exp':<12}" + "".join(f"{c:>18}" for c in COLUMNS))
    for r in rows:
        print(f"{r['exp']:<12}" + "".join(f"{r[c]:>18.4f}" if r[c] is not None else f"{'-':>18}" for c in COLUMNS))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Guidance-scale ablation: for each scale, generate the 100 test clips and evaluate them.
#   ./run_ablation.sh <gpu id> <scale> [<scale> ...]      e.g. ./run_ablation.sh 0 1 3 5
# Everything else (100 steps, 10 s, seed 0) matches the baseline in generated_samples/ (scale 7).
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=$1
shift
H=$(git rev-parse --short HEAD)

for S in "$@"; do
    EXP=scale_$S
    echo "=== $EXP ($(date +%H:%M)) ==="
    .venv/bin/python generate_batch.py --scale "$S" --out_dir outputs/$EXP > logs/generate_${EXP}_$H.log 2>&1
    ./run_eval.sh outputs/$EXP $EXP > logs/eval_${EXP}_$H.log 2>&1
    echo "=== $EXP done ($(date +%H:%M)) ==="
done
echo "ALL DONE"

#!/usr/bin/env bash
# APG runs: for each <name>:<scale>:<momentum> spec, generate the 100 test clips with APG (eta 0) and evaluate.
#   ./run_apg.sh <gpu id> apg_s7:7:-0.5 apgproj_s7:7:0
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=$1
shift
H=$(git rev-parse --short HEAD)

for SPEC in "$@"; do
    IFS=: read -r EXP S M <<< "$SPEC"
    echo "=== $EXP scale=$S momentum=$M ($(date +%H:%M)) ==="
    .venv/bin/python generate_batch.py --guidance apg --eta 0 --scale "$S" --momentum "$M" \
        --out_dir outputs/$EXP > logs/generate_${EXP}_$H.log 2>&1
    ./run_eval.sh outputs/$EXP $EXP > logs/eval_${EXP}_$H.log 2>&1
    echo "=== $EXP done ($(date +%H:%M)) ==="
done
echo "ALL DONE"

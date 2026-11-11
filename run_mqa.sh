#!/usr/bin/env bash
# MQAScore for a list of experiments on one GPU: ./run_mqa.sh <gpu> name:audio_dir[:shift] ...
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1
shift
H=$(git rev-parse --short HEAD)
for SPEC in "$@"; do
    IFS=: read -r EXP DIR SHIFT <<< "$SPEC"
    echo "=== $EXP ($(date +%H:%M)) ==="
    .venv-eval/bin/python mqascore.py --exp "$EXP" --audio_dir "$DIR" --shift "${SHIFT:-0}" > logs/mqa_${EXP}_$H.log 2>&1
    tail -1 logs/mqa_${EXP}_$H.log
done
echo "ALL DONE"

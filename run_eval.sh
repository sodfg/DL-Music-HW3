#!/usr/bin/env bash
# Evaluate one set of generated clips with av-benchmark (FD-VGG, KL-PaSST, LAION-CLAP, ...).
#   ./run_eval.sh <generated mp3 dir> <experiment name>      e.g. ./run_eval.sh generated_samples baseline
set -euo pipefail
cd "$(dirname "$0")"
PRED=$1
EXP=$2
EVAL=outputs/eval
PY=$PWD/.venv-eval/bin/python
AVB=external/av-benchmark

# The HW requires the music_audioset_epoch_15_esc_90.14.pt CLAP checkpoint; av-benchmark hard-codes 89.98.
sed -i 's/music_speech_audioset_epoch_15_esc_89.98.pt/music_audioset_epoch_15_esc_90.14.pt/' \
    $AVB/extract_text.py $AVB/av_bench/extraction_models.py
grep -q music_audioset_epoch_15_esc_90.14.pt $AVB/av_bench/extraction_models.py
# extract_text.py resolves weights/ one level above the repo (external/weights), so point that at the same folder.
[ -e external/weights ] || ln -s av-benchmark/weights external/weights
# Loaded unconditionally by av-benchmark even with --skip_video_related (README step 4).
[ -f $AVB/weights/synchformer_state_dict.pth ] || wget -nv -O $AVB/weights/synchformer_state_dict.pth \
    https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth

"$PY" prepare_eval.py --pred "$PRED" --exp "$EXP" --eval_dir $EVAL

# Caption embeddings, shared by all experiments
[ -f $EVAL/gt_cache/clap_laion_text.pth ] || \
    (cd $AVB && "$PY" extract_text.py --text_csv ../../$EVAL/text.csv --output_cache_path ../../$EVAL/gt_cache)

(cd $AVB && "$PY" evaluate.py --gt_audio ../../$EVAL/gt_wav --gt_cache ../../$EVAL/gt_cache \
    --pred_audio ../../$EVAL/$EXP/pred_wav --pred_cache ../../$EVAL/$EXP/pred_cache \
    --audio_length 10 --skip_video_related --num_workers 8)

mkdir -p results
cp $EVAL/$EXP/pred_cache/output_metrics.json results/metrics_$EXP.json
cat results/metrics_$EXP.json

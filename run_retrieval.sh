#!/usr/bin/env bash
# Build the eval env (av-benchmark + LAION-CLAP), fetch the CLAP 90.14 checkpoint, then run retrieval.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")"
CKPT=external/av-benchmark/weights/music_audioset_epoch_15_esc_90.14.pt

[ -x .venv-eval/bin/python ] || uv venv --python 3.11 .venv-eval
uv pip install --python .venv-eval/bin/python torch==2.11.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-eval/bin/python -e external/av-benchmark
mkdir -p "$(dirname "$CKPT")"
wget -nv -c -O "$CKPT" https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt

CUDA_VISIBLE_DEVICES=0 .venv-eval/bin/python retrieval.py --ckpt "$CKPT"
echo "ALL DONE"

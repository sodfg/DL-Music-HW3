#!/usr/bin/env bash
# Build the eval env (av-benchmark + LAION-CLAP), fetch the CLAP 90.14 checkpoint, then run retrieval.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")"
CKPT=external/av-benchmark/weights/music_audioset_epoch_15_esc_90.14.pt

[ -x .venv-eval/bin/python ] || uv venv --python 3.11 .venv-eval
# torchaudio>=2.9 needs torchcodec+FFmpeg to load audio; av-benchmark uses torchaudio.load
uv pip install --python .venv-eval/bin/python torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-eval/bin/python -e external/av-benchmark
uv pip install --python .venv-eval/bin/python "transformers<5"  # MS-CLAP needs tokenizer.encode_plus (removed in 5.x)
mkdir -p "$(dirname "$CKPT")"
wget -nv -c -O "$CKPT" https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt

CUDA_VISIBLE_DEVICES=0 .venv-eval/bin/python retrieval.py --ckpt "$CKPT"
echo "ALL DONE"

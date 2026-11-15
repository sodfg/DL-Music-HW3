# DL in Music HW3: Text-to-Music Generation and Evaluation

B12901157. Stable Audio Open 1.0 with classifier-free guidance (CFG), CLAP retrieval of MusicCaps references,
evaluation with av-benchmark, a guidance-scale ablation, Adaptive Projected Guidance (APG) as the improvement method,
and an MQAScore-style LALM judge.

## Results (100 test captions, 100 steps, 10 s, seed 0)

| Setting | FD-VGG ↓ | KL-PaSST ↓ | LAION-CLAP ↑ | Samples > 1.0 (%) | MQAScore ↑ | CaptionQA ↑ |
|---|---|---|---|---|---|---|
| CFG w=1 | 3.94 | 1.196 | 0.251 | 0.0002 | 0.944 | 0.777 |
| CFG w=7 (submitted) | 2.20 | 1.061 | 0.324 | 0.057 | 0.969 | 0.874 |
| CFG w=15 | 2.21 | 0.960 | 0.319 | 0.287 | 0.970 | 0.881 |
| APG w=7, β=−0.5 | 2.28 | 1.011 | 0.315 | 0.020 | 0.965 | 0.864 |
| APG w=15, β=−0.5 | 2.12 | 1.001 | 0.323 | 0.159 | 0.968 | 0.889 |

All settings (w = 0, 1, 3, 5, 7, 10, 15 and four APG variants) are in `results/summary.csv`,
`results/clip_stats.csv` and `results/mqa_*.json`. The analysis is in the report.

## Setup

Hardware used: one RTX A5000 (24 GB). Generation takes about 10 s per clip in fp16 (about 17 min for 100 clips).

Two Python 3.11 environments are used, because the generation stack (torch 2.11, transformers 5) and the
evaluation stack (av-benchmark: torchaudio.load without torchcodec, MS-CLAP needs transformers < 5) conflict.

```bash
# Generation env
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt

# Evaluation env: run_retrieval.sh builds it (clones av-benchmark at f351b9a, installs requirements-eval.txt
# equivalents, downloads music_audioset_epoch_15_esc_90.14.pt) and then runs the retrieval.
```

`stabilityai/stable-audio-open-1.0` is gated on Hugging Face: accept its license and run `huggingface-cli login`.
MQAScore uses `Qwen/Qwen2-Audio-7B-Instruct` (about 16 GB in bf16).

Data (not in this repository):

- `musiccaps/audio_mp3/<youtube_id>.mp3`: the 5,367 MusicCaps clips provided by the course
- `sample_code/test.csv`: the 100 test captions (`id,caption`)

## Reproduce

```bash
# 1. CFG check against diffusers (max difference 0)
.venv/bin/python sample_code/generate.py --prompt "jazz, electric guitar, piano, relaxing" --out out.wav --check

# 2. Retrieval: builds .venv-eval and writes retrieval.json (+ retrieval.top5.csv)
./run_retrieval.sh

# 3. Generation: the submitted samples (CFG, w = 7)
.venv/bin/python generate_batch.py --out_dir generated_samples

# 4. Evaluation with av-benchmark -> results/metrics_baseline.json
./run_eval.sh generated_samples baseline

# 5. Guidance-scale ablation: generate + evaluate each scale on a GPU
./run_ablation.sh 0 1 3 5
./run_ablation.sh 1 10 15 0

# 6. APG (eta = 0): name:scale:momentum
./run_apg.sh 0 apg_s7:7:-0.5 apgproj_s7:7:0
./run_apg.sh 1 apg_s15:15:-0.5 apgproj_s15:15:0

# 7. Summary table and clipping statistics
.venv-eval/bin/python summarize.py
.venv/bin/python clip_stats.py outputs/scale_1:cfg_s1 generated_samples:cfg_s7 outputs/scale_15:cfg_s15 \
    outputs/apg_s7:apg_s7 outputs/apg_s15:apg_s15

# 8. MQAScore (name:wav_dir[:caption_shift]); the first run extracts concepts into results/mqa_concepts.json
./run_mqa.sh 0 gt:outputs/eval/gt_wav baseline:outputs/eval/baseline/pred_wav gt_shift:outputs/eval/gt_wav:37
```

Logs go to `logs/<exp>_<commit>.log`.

## Files

| File | Purpose |
|---|---|
| `sample_code/cfg.py` | CFG: `uncond + w * (cond - uncond)` |
| `sample_code/generate.py` | Course sampling loop that calls `cfg()` (unchanged) |
| `retrieval.py`, `run_retrieval.sh` | LAION-CLAP (90.14 checkpoint) text-to-audio retrieval over MusicCaps |
| `generate_batch.py` | One clip per test caption, `--guidance cfg` or `apg` |
| `apg.py` | APG (Sadat et al., ICLR 2025) applied to the denoised estimate, mapped back to v |
| `prepare_eval.py`, `run_eval.sh` | Lay out wav files for av-benchmark and run it (switches its CLAP checkpoint to 90.14) |
| `run_ablation.sh`, `run_apg.sh` | Generate and evaluate several settings in a row |
| `summarize.py` | Collect `results/metrics_*.json` into `results/summary.csv` |
| `clip_stats.py` | Peak, RMS and share of samples above 1.0 per setting |
| `mqascore.py`, `run_mqa.sh` | MQAScore-style judge: concept extraction and P(yes) per concept with Qwen2-Audio |
| `retrieval.json` | Caption id → retrieved MusicCaps id |
| `results/` | All metrics, clipping statistics and MQAScore outputs |

## Notes

- MQAScore has no public code, so it is implemented from the course description (LALM judge, concept level,
  confidence). Concepts for captions 83 and 97 were split by hand because the model returned them on one line.
  The judge runs in bf16; fp16 overflowed to NaN.
- Captions were not used to tune anything; every setting uses the same seed and sampler.

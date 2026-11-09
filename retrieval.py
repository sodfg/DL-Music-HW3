"""CLAP text-to-music retrieval: for each test caption, find the closest MusicCaps clip.

Uses LAION-CLAP (HTSAT-base) with the music_audioset_epoch_15_esc_90.14.pt checkpoint required by the HW.

    python retrieval.py --ckpt external/av-benchmark/weights/music_audioset_epoch_15_esc_90.14.pt
"""

import argparse
import csv
import json
from pathlib import Path

import librosa
import laion_clap
import numpy as np
import torch
from tqdm import tqdm

SR = 48000          # LAION-CLAP expects 48 kHz mono
N_SAMPLES = 10 * SR  # 10-second input window


def load_clip(path):
    """Load a clip as 48 kHz mono, cropped (or repeat-padded) to exactly 10 s, so embeddings are deterministic."""
    wav, _ = librosa.load(path, sr=SR, mono=True)
    if len(wav) < N_SAMPLES:
        wav = np.tile(wav, N_SAMPLES // len(wav) + 1)
    wav = wav[:N_SAMPLES]
    # Same int16 round-trip as the LAION-CLAP README, to match the training-time preprocessing.
    return (np.clip(wav, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32767


@torch.no_grad()
def embed_audio(model, paths, batch_size):
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="audio"):
        batch = np.stack([load_clip(p) for p in paths[i:i + batch_size]])
        embs.append(model.get_audio_embedding_from_data(x=batch, use_tensor=False))
    return np.concatenate(embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--audio_dir", default="musiccaps/audio_mp3")
    parser.add_argument("--captions", default="sample_code/test.csv")
    parser.add_argument("--out", default="retrieval.json")
    parser.add_argument("--cache", default="outputs/clap_musiccaps_audio.npz")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
    model.load_ckpt(args.ckpt, verbose=False)

    # 1. MusicCaps audio embeddings (slow part, cached to disk)
    cache = Path(args.cache)
    if cache.exists():
        data = np.load(cache)
        names, audio_emb = list(data["names"]), data["emb"]
    else:
        paths = sorted(Path(args.audio_dir).glob("*.mp3"))
        names = [p.stem for p in paths]  # file stem = MusicCaps YouTube id
        audio_emb = embed_audio(model, [str(p) for p in paths], args.batch_size)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, names=np.array(names), emb=audio_emb)
    print(f"{len(names)} MusicCaps clips")

    # 2. Caption embeddings
    with open(args.captions, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["id"] for r in rows]
    text_emb = model.get_text_embedding([r["caption"] for r in rows], use_tensor=False)
    print(f"{len(ids)} captions")

    # 3. Cosine similarity, take the best match for each caption
    audio_emb = audio_emb / np.linalg.norm(audio_emb, axis=1, keepdims=True)
    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    sim = text_emb @ audio_emb.T  # (n_captions, n_clips)
    best = sim.argmax(axis=1)

    retrieval = {i: names[j] for i, j in zip(ids, best)}
    with open(args.out, "w") as f:
        json.dump(retrieval, f, indent=2)

    # Top-5 per caption with scores, for inspection and the report
    top5 = np.argsort(-sim, axis=1)[:, :5]
    with open(Path(args.out).with_suffix(".top5.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "rank", "musiccaps_id", "cosine"])
        for row, i in enumerate(ids):
            for rank, j in enumerate(top5[row], 1):
                w.writerow([i, rank, names[j], f"{sim[row, j]:.4f}"])

    print(f"Saved {args.out}; best cosine mean {sim.max(axis=1).mean():.4f}, "
          f"{len(set(retrieval.values()))} unique clips for {len(ids)} captions")


if __name__ == "__main__":
    main()

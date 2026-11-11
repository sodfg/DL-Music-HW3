"""MQAScore-style text-music alignment with an audio language model as judge (Qwen2-Audio-7B-Instruct).

Following the idea of AQAScore / MQAScore (LALM as a judge, fine-grained concepts, confidence):
  1. Concept extraction (once, cached): the same LALM, text-only, splits each caption into short musical
     concepts (genre, instruments, mood, tempo, vocals, ...)  -> results/mqa_concepts.json
  2. For each clip and each concept ask "Does this music contain <concept>? Answer yes or no." and take the
     model's confidence P(yes) = p(Yes) / (p(Yes) + p(No)) from the logits of the first answer token.
     MQAScore(clip) = mean P(yes) over its concepts; the experiment score is the mean over the 100 clips.
     The LALM has a strong yes-bias (P(yes) is high even for mismatched pairs), so we also keep the raw
     log p(Yes) as in AQAScore, and score mismatched pairs (--shift) as a lower bound.
  3. Also a caption-level score (one question with the whole caption, AQAScore-style) for comparison.

    python mqascore.py --concepts_only
    python mqascore.py --exp baseline --audio_dir outputs/eval/baseline/pred_wav
    python mqascore.py --exp gt --audio_dir outputs/eval/gt_wav
    python mqascore.py --exp baseline_shift --audio_dir outputs/eval/baseline/pred_wav --shift 1   # mismatched pairs
"""

import argparse
import csv
import json
import re
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
CONCEPTS = Path("results/mqa_concepts.json")
EXTRACT_PROMPT = (
    "Split the following music description into its distinct musical concepts, such as genre, instruments, "
    "vocals, mood, tempo or rhythm, and sound quality. Write each concept as a short phrase (2 to 6 words) "
    "that can be checked by listening, one per line, with no numbering and no extra text. Only list concepts "
    "that the description actually states.\n\n"
    "Description: An upbeat reggae song with a male singer, offbeat guitar chords and a warm bass line, "
    "with a sunny, carefree mood.\n"
    "Concepts:\nreggae music\nupbeat tempo\nmale vocals\noffbeat guitar chords\nwarm bass line\n"
    "sunny, carefree mood\n\n"
    "Description: {caption}\nConcepts:"
)
# Replies the model sometimes adds that are not checkable concepts
FILLER = ("not specified", "not mentioned", "not described", "no specific", "not provided", "no particular",
          "may vary", "not highlighted", "but not")
CONCEPT_Q = "Does this music contain {concept}? Answer yes or no."
CAPTION_Q = "Does this music match the following description? {caption} Answer yes or no."


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL)
    processor.tokenizer.padding_side = "left"  # so position -1 is the answer position for every row
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.eval()
    return processor, model


def chat(processor, text, with_audio):
    content = ([{"type": "audio", "audio_url": "clip"}] if with_audio else []) + [{"type": "text", "text": text}]
    return processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                         tokenize=False)


@torch.no_grad()
def extract_concepts(processor, model, rows):
    concepts = {}
    for row in rows:
        prompt = chat(processor, EXTRACT_PROMPT.format(caption=row["caption"]), with_audio=False)
        inputs = processor(text=prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        reply = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        lines = reply.replace("\\n", "\n").splitlines()
        items = [re.sub(r"^\s*\d+[.)]\s*", "", line).strip(" -*•\t").rstrip(".") for line in lines]
        concepts[row["id"]] = [c for c in items if c and not any(f in c.lower() for f in FILLER)][:8]
        print(row["id"], concepts[row["id"]], flush=True)
    return concepts


def answer_token_ids(tokenizer):
    ids = lambda words: sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in words})
    return ids(["Yes", "yes", " Yes", " yes"]), ids(["No", "no", " No", " no"])


@torch.no_grad()
def p_yes(processor, model, audio, questions, yes_ids, no_ids):
    """(P(yes | yes or no), log p(yes)) for each question about the same clip, in one padded batch."""
    texts = [chat(processor, q, with_audio=True) for q in questions]
    inputs = processor(text=texts, audio=[audio] * len(texts), sampling_rate=16000, return_tensors="pt",
                       padding=True).to("cuda")
    inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
    logits = model(**inputs).logits[:, -1, :].float()
    probs = logits.softmax(-1)
    yes, no = probs[:, yes_ids].sum(-1), probs[:, no_ids].sum(-1)
    return (yes / (yes + no)).tolist(), yes.log().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp")
    parser.add_argument("--audio_dir", help="directory of <id>.wav (or .mp3)")
    parser.add_argument("--captions", default="sample_code/test.csv")
    parser.add_argument("--shift", type=int, default=0, help="pair clip i with caption i+shift (sanity check)")
    parser.add_argument("--concepts_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="only the first N clips (smoke test)")
    args = parser.parse_args()

    with open(args.captions, newline="") as f:
        rows = list(csv.DictReader(f))
    processor, model = load_model()

    if not CONCEPTS.exists():
        CONCEPTS.parent.mkdir(exist_ok=True)
        CONCEPTS.write_text(json.dumps(extract_concepts(processor, model, rows), indent=1, ensure_ascii=False))
    concepts = json.loads(CONCEPTS.read_text())
    if args.concepts_only:
        return

    yes_ids, no_ids = answer_token_ids(processor.tokenizer)
    per_clip = {}
    for n, row in enumerate(rows[: args.limit]):
        q_row = rows[(n + args.shift) % len(rows)]  # caption the clip is judged against
        path = next(Path(args.audio_dir).glob(f"{row['id']}.*"))
        audio, _ = librosa.load(path, sr=16000, mono=True)
        cs = concepts[q_row["id"]]
        scores, logp = p_yes(processor, model, audio,
                       [CONCEPT_Q.format(concept=c) for c in cs] + [CAPTION_Q.format(caption=q_row["caption"])],
                       yes_ids, no_ids)
        per_clip[row["id"]] = {"concepts": dict(zip(cs, scores[:-1])), "mqa": float(np.mean(scores[:-1])),
                               "caption_level": scores[-1],
                               "mqa_logp": float(np.mean(logp[:-1])), "caption_logp": logp[-1]}
        print(f"[{n + 1}] {row['id']} mqa={per_clip[row['id']]['mqa']:.3f} "
              f"caption={scores[-1]:.3f}", flush=True)

    summary = {"exp": args.exp, "n": len(per_clip),
               "MQAScore": float(np.mean([v["mqa"] for v in per_clip.values()])),
               "CaptionQA": float(np.mean([v["caption_level"] for v in per_clip.values()])),
               "MQA_logp": float(np.mean([v["mqa_logp"] for v in per_clip.values()])),
               "Caption_logp": float(np.mean([v["caption_logp"] for v in per_clip.values()]))}
    print(summary)
    if args.limit is None:
        Path(f"results/mqa_{args.exp}.json").write_text(json.dumps({"summary": summary, "per_clip": per_clip},
                                                                  indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

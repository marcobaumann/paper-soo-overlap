"""
train_scenario_sham.py — SCENARIO-EXPOSURE SHAM (control ladder, Rung 2).

Question this arm answers: is the honesty from the SOO *objective*, or just from
the model *seeing these scenarios*? This is the objection a skeptic raises after
SOO beats the wikitext sham ("sure, but would it beat just training on the
prompts?"). It holds the DATA constant and varies ONLY the objective.

Difference from the SOO arm (../01_paper_reproduction/train.py):
    SOO arm:  two forward passes on (self, other) PAIRS,
              loss = MSE(A_self, A_other)   [activation-matching]
    This arm: one forward pass on the SAME prompt strings, flattened, no pairing,
              loss = causal-LM next-token loss   [ordinary imitation of the text]

What is held constant so the comparison is clean:
    * SAME text: we train on exactly the strings SOO sees (build_training_pairs),
      flattened into a single list of self-strings + other-strings.
    * NO answers: prompts only. We never append an honest/deceptive room, so the
      model is never shown the "right" answer. This is pure exposure, not
      supervised honesty. (Showing answers would be a different experiment.)
    * SAME model / quant / LoRA / lr / batch / layers (shared model_utils.py).
    * SAME dose: match the SOO arm's exact optimizer-step count per seed.

Usage:
    python train_scenario_sham.py --seed 0 --out ./checkpoints/scen_seed0 \
        --soo_steps_json ../01_paper_reproduction/checkpoints/soo_seed0/steps_taken.json
"""

import argparse, json, os, random
import numpy as np
import torch

from config import MODEL, TRAIN
from data import build_training_pairs
from model_utils import load_model_and_tokenizer


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def target_steps(seed, soo_steps_json):
    """Match the SOO arm's optimizer-step count exactly (dose matching)."""
    if soo_steps_json and os.path.exists(soo_steps_json):
        return json.load(open(soo_steps_json))["steps"]
    n_pairs = len(build_training_pairs(seed))
    per_epoch = -(-n_pairs // TRAIN.batch_size)  # ceil div
    return TRAIN.epochs * per_epoch


def build_exposure_texts(seed):
    """
    Flatten the SOO training pairs into a single list of prompt strings.
    These are the EXACT strings SOO trains on — self-prompts and other-prompts —
    but here treated as plain text, unpaired, with no answers.
    """
    pairs = build_training_pairs(seed)
    texts = []
    for p in pairs:
        texts.append(p["self"])
        texts.append(p["other"])
    return texts


def batches(tokenizer, texts, device, batch_size, n_steps, seed):
    """Cycle through the exposure texts, yielding LM batches until n_steps reached."""
    rng = random.Random(seed)
    pool = texts[:]
    rng.shuffle(pool)
    idx, produced = 0, 0
    while produced < n_steps:
        if idx + batch_size > len(pool):
            rng.shuffle(pool); idx = 0
        chunk = pool[idx: idx + batch_size]; idx += batch_size
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()  # next-token LM loss on the prompt text
        yield enc
        produced += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--soo_steps_json", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda"
    model, tokenizer = load_model_and_tokenizer(MODEL, TRAIN)  # SAME loading as SOO arm
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=TRAIN.lr)

    n_steps = target_steps(args.seed, args.soo_steps_json)
    texts = build_exposure_texts(args.seed)
    print(f"[scenario-sham] {len(texts)} exposure strings, matching SOO dose: {n_steps} steps")

    model.train()
    steps = 0
    for batch in batches(tokenizer, texts, device, TRAIN.batch_size, n_steps, args.seed):
        loss = model(**batch).loss
        opt.zero_grad(); loss.backward(); opt.step()
        steps += 1
        if steps % 20 == 0:
            print(f"step {steps}/{n_steps} lm_loss {loss.item():.4f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "steps_taken.json"), "w") as f:
        json.dump({"steps": steps, "arm": "scenario_exposure", "seed": args.seed}, f, indent=2)
    print(f"[done] scenario-sham seed {args.seed}: {steps} steps -> {args.out}")


if __name__ == "__main__":
    main()

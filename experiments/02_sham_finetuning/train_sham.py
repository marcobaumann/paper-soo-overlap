"""
train_sham.py — SHAM CONTROL arm.

THE ONLY DIFFERENCE FROM ../01_paper_reproduction/train.py IS THE TRAINING SIGNAL.

  Reproduction arm: loss = MSE(A_self, A_other)   (SOO)  on burglar prompt pairs.
  Sham arm (this):  loss = causal-LM next-token loss on UNRELATED text.

Everything else is held constant so the SOO-vs-sham gap is attributable to the
SOO loss and nothing else:
  - same model, quantization, LoRA config (via shared model_utils.py)
  - same layers touched (q/v LoRA)
  - same optimizer, lr, batch size
  - same DOSE: we match the exact number of optimizer steps the SOO arm took,
    read from ../01_paper_reproduction/checkpoints/soo_seed{seed}/steps_taken.json
    (falls back to the analytic step count if that file isn't present).

"Unrelated text" = wikitext-103 raw. Nothing about deception, self/other, rooms,
or Bob. This isolates "generic fine-tuning of the same magnitude."

Usage:
    python train_sham.py --seed 0 --out ./checkpoints/sham_seed0 \
        --soo_steps_json ../01_paper_reproduction/checkpoints/soo_seed0/steps_taken.json
"""

import argparse, json, os, random
import numpy as np
import torch
from datasets import load_dataset

from config import MODEL, TRAIN
from data import build_training_pairs
from model_utils import load_model_and_tokenizer


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def target_steps(seed, soo_steps_json):
    """Match the SOO arm's optimizer-step count exactly (dose matching)."""
    if soo_steps_json and os.path.exists(soo_steps_json):
        return json.load(open(soo_steps_json))["steps"]
    # Fallback: analytic count = epochs * ceil(n_pairs / batch)
    n_pairs = len(build_training_pairs(seed))
    per_epoch = -(-n_pairs // TRAIN.batch_size)  # ceil div
    return TRAIN.epochs * per_epoch


def unrelated_text_batches(tokenizer, device, batch_size, n_batches, seed):
    """Stream wikitext lines into fixed-size LM batches."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    buf, produced = [], 0
    for row in ds:
        t = row["text"].strip()
        if len(t) < 40:  # skip empties/headers
            continue
        buf.append(t)
        if len(buf) == batch_size:
            enc = tokenizer(buf, return_tensors="pt", padding=True,
                            truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            enc["labels"] = enc["input_ids"].clone()
            yield enc
            buf, produced = [], produced + 1
            if produced >= n_batches:
                return


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
    print(f"[sham] matching SOO dose: {n_steps} optimizer steps")

    model.train()
    steps = 0
    while steps < n_steps:
        for batch in unrelated_text_batches(tokenizer, device, TRAIN.batch_size,
                                            n_steps - steps, args.seed + steps):
            out = model(**batch)          # standard next-token LM loss
            loss = out.loss
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1
            if steps % 20 == 0:
                print(f"step {steps}/{n_steps} lm_loss {loss.item():.4f}")
            if steps >= n_steps:
                break

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "steps_taken.json"), "w") as f:
        json.dump({"steps": steps, "arm": "sham", "seed": args.seed}, f, indent=2)
    print(f"[done] sham seed {args.seed}: {steps} steps -> {args.out}")


if __name__ == "__main__":
    main()

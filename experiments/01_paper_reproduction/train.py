"""
train.py — SOO fine-tuning (the reproduction arm).

Runs the paper's Mistral-7B recipe: LoRA on q/v, 15 epochs, lr 1e-4, batch 4,
loss = MSE(A_self, A_other) at o_proj layer 19, mean-pooled.

Usage:
    python train.py --seed 0 --out ./checkpoints/soo_seed0

The number of optimizer steps this arm takes is written to steps_taken.json so
the sham arm can match the dose exactly.
"""

import argparse, json, os, random
import numpy as np
import torch

from config import MODEL, TRAIN
from data import build_training_pairs
from model_utils import load_model_and_tokenizer
from soo import OProjCapture, soo_loss


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def tokenize_side(tokenizer, texts, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    return {k: v.to(device) for k, v in enc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda"
    model, tokenizer = load_model_and_tokenizer(MODEL, TRAIN)
    capture = OProjCapture(model, MODEL.soo_layer)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=TRAIN.lr)

    pairs = build_training_pairs(args.seed)
    bs = TRAIN.batch_size
    steps = 0
    model.train()
    for epoch in range(TRAIN.epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            self_b = tokenize_side(tokenizer, [b["self"] for b in batch], device)
            other_b = tokenize_side(tokenizer, [b["other"] for b in batch], device)

            loss = soo_loss(model, capture, self_b, other_b, TRAIN.pooling)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1
            if steps % 20 == 0:
                print(f"epoch {epoch} step {steps} soo_loss {loss.item():.3e}")

    capture.remove()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "steps_taken.json"), "w") as f:
        json.dump({"steps": steps, "pooling": TRAIN.pooling, "seed": args.seed}, f, indent=2)
    print(f"[done] seed {args.seed}: {steps} optimizer steps -> {args.out}")


if __name__ == "__main__":
    main()

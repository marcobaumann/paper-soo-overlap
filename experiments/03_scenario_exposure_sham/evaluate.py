"""
evaluate.py — Runs a checkpoint on the held-out test scenarios and dumps every
raw response for offline classification. Also reports Latent SOO (MSE).

Classification is NOT done here. An earlier version classified locally by
substring-matching room names in the response, but that misclassifies any
response that mentions both rooms as "unclear" — which turned out to be
common: the model often states its recommendation first, then notes the true
location afterward as a rationale (e.g. "...point to the dining room... while
the platinum bracelet is actually in the utility room"). Instead, every
response is dumped with "classification": null; run classify_responses.py
(locally, not on the training pod — keep the Anthropic API key off a rented
instance) to fill it in via an LLM judge, then aggregate.py for the summary.

Usage:
    python evaluate.py --ckpt ./checkpoints/soo_seed0 --seed 0 --tag soo
"""

import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import MODEL, TRAIN, EVAL
from data import build_test_scenarios, build_latent_soo_pairs
from soo import OProjCapture, measure_latent_soo


def load_ckpt(ckpt, device):
    tok = AutoTokenizer.from_pretrained(ckpt)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL.name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ckpt).to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True, help="label for this arm, e.g. soo/sham/baseline")
    args = ap.parse_args()

    device = "cuda"
    model, tok = load_ckpt(args.ckpt, device)

    scenarios = build_test_scenarios(EVAL.n_test_examples, args.seed)
    responses = []
    for s in scenarios:
        prompt = s["prompt"] + MODEL.response_primer
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(device)
        attention_mask = torch.ones_like(ids)
        out = model.generate(
            ids, attention_mask=attention_mask,
            max_new_tokens=EVAL.max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        resp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        responses.append({
            "prompt": s["prompt"], "honest_room": s["honest_room"],
            "deceptive_room": s["deceptive_room"],
            "response": resp, "classification": None,
        })

    n = len(scenarios)

    capture = OProjCapture(model, MODEL.soo_layer)
    latent = measure_latent_soo(
        model, capture, tok,
        build_latent_soo_pairs(EVAL.n_latent_soo_pairs, args.seed),
        TRAIN.pooling, device,
    )
    capture.remove()

    result = {
        "tag": args.tag, "seed": args.seed, "n": n,
        "latent_soo_mse": latent, "pooling": TRAIN.pooling,
        "responses": responses,
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/{args.tag}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {args.tag} seed {args.seed}: {n} responses -> {path} (classification pending)")


if __name__ == "__main__":
    main()

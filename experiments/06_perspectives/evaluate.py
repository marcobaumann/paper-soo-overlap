"""
evaluate.py — PERSPECTIVES TEST. Evaluates a model on the Perspectives
theory-of-mind scenario and dumps every raw response for offline classification.
No training, no Latent SOO — this is a pure behavioral eval of whether the model
still distinguishes its own knowledge from Bob's.

Works on either:
  - a SOO-trained checkpoint (--ckpt ../01_paper_reproduction/checkpoints/soo_seedN)
  - the untrained base model (--base)

so we can reproduce the paper's full "100% -> 100%" Perspectives claim ourselves
(base accuracy vs SOO-FT accuracy). If our collapse hypothesis is right, the
SOO-trained model's accuracy should drop below the base model's — direct
evidence the self/other distinction was erased, which MT-Bench (general
capability) would NOT catch.

Classification is NOT done here (same pattern as 01/05): every response is
dumped with "classification": null; run classify_perspectives.py locally to
judge correct/incorrect/unclear, then aggregate.py for the accuracy summary.

Usage:
    python evaluate.py --ckpt ../01_paper_reproduction/checkpoints/soo_seed0 --seed 0 --tag persp_soo
    python evaluate.py --base --seed 0 --tag persp_base
"""

import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import MODEL, EVAL
from data import build_perspectives_scenarios


def load_ckpt(ckpt, device):
    tok = AutoTokenizer.from_pretrained(ckpt)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL.name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ckpt).to(device)
    model.eval()
    return model, tok


def load_base(device):
    tok = AutoTokenizer.from_pretrained(MODEL.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL.name, torch_dtype=torch.bfloat16, device_map="auto"
    ).to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="LoRA checkpoint dir; omit with --base")
    ap.add_argument("--base", action="store_true", help="evaluate the untrained base model")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True, help="label, e.g. persp_soo / persp_base")
    args = ap.parse_args()

    if not args.base and not args.ckpt:
        ap.error("provide --ckpt <dir> or --base")

    device = "cuda"
    model, tok = load_base(device) if args.base else load_ckpt(args.ckpt, device)

    scenarios = build_perspectives_scenarios(EVAL.n_test_examples, args.seed)
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
            "prompt": s["prompt"], "self_room": s["self_room"],
            "bob_room": s["bob_room"],
            "response": resp, "classification": None,
        })

    n = len(scenarios)
    result = {
        "tag": args.tag, "seed": args.seed, "n": n,
        "responses": responses,
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/{args.tag}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {args.tag} seed {args.seed}: {n} responses -> {path} (classification pending)")


if __name__ == "__main__":
    main()

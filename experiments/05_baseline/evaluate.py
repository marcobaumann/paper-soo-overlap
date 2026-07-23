"""
evaluate.py — BASELINE VARIANT. Evaluates the UNTRAINED Mistral-7B base model
(no LoRA adapter, no SOO fine-tuning) on the held-out test scenarios, and dumps
every raw response for offline classification. Also reports Latent SOO (MSE),
both at the would-be-trained layer and averaged across all MLP layers.

This is the "before" number that pairs with ../01_paper_reproduction's "after"
— run on the EXACT same 250-scenario test sets (same expanded_inventory.json,
same per-seed sampling), so the deceptive-rate and Latent-SOO deltas are
apples-to-apples on our own setup rather than borrowed from the paper. The
paper's own Mistral baseline is 73.6% deceptive / Latent SOO 0.107.

The ONLY difference from ../01_paper_reproduction/evaluate.py is load_base()
vs load_ckpt(): here we load the raw base model with no adapter. Everything
downstream (scenario building, generation, both Latent SOO measurements,
response dumping) is identical, so the numbers are directly comparable.

Classification is NOT done here (same as 01): every response is dumped with
"classification": null; run classify_responses.py locally to fill it in via an
LLM judge, then aggregate.py for the summary.

Usage:
    python evaluate.py --seed 0 --tag baseline
"""

import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL, TRAIN, EVAL
from data import build_test_scenarios, build_latent_soo_pairs
from soo import OProjCapture, AllLayersCapture, measure_latent_soo, measure_latent_soo_all_layers


def load_base(device):
    """Load the untrained base model + tokenizer, no LoRA adapter. Mirrors 01's
    load_ckpt loading (bf16, device_map=auto) minus the PeftModel wrap, so the
    base numbers are comparable to 01's fine-tuned numbers. pad_token is set to
    eos_token to match how model_utils.py (and thus 01's saved checkpoints)
    configure the tokenizer."""
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
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="baseline", help="label for this arm")
    args = ap.parse_args()

    device = "cuda"
    model, tok = load_base(device)

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
    latent_pairs = build_latent_soo_pairs(EVAL.n_latent_soo_pairs, args.seed)

    trained_layer_capture = OProjCapture(model, MODEL.soo_layer)
    latent_trained_layer = measure_latent_soo(
        model, trained_layer_capture, tok, latent_pairs, TRAIN.pooling, device,
    )
    trained_layer_capture.remove()

    all_layers_capture = AllLayersCapture(model)
    latent_all_layers = measure_latent_soo_all_layers(
        model, all_layers_capture, tok, latent_pairs, TRAIN.pooling, device,
    )
    all_layers_capture.remove()

    result = {
        "tag": args.tag, "seed": args.seed, "n": n,
        "latent_soo_mse": latent_all_layers,
        "latent_soo_trained_layer_mse": latent_trained_layer,
        "pooling": TRAIN.pooling,
        "responses": responses,
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/{args.tag}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] {args.tag} seed {args.seed}: {n} responses -> {path} (classification pending)")


if __name__ == "__main__":
    main()

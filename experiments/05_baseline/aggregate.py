"""
aggregate.py — Collapse per-seed result JSONs into mean +/- SD, paper-style.

Classification comes from classify_responses.py (an LLM judge run locally),
stored per response in each results/{tag}_seed{N}.json's "responses" list —
this script just counts labels and reports latent_soo_mse mean/SD.

latent_soo_mse (mean across all MLP layers -- the paper's Table 4 methodology
for Mistral-7B specifically; attention layers only enter the paper's
methodology as a Gemma-2-27B-it-specific fallback) is the number comparable
to the paper's Table 4. latent_soo_trained_layer_mse (the single attention
layer the SOO loss directly optimizes) is expected to collapse toward zero
given enough training almost tautologically -- it's a training sanity check,
not a paper-comparable number.

Usage:
    python aggregate.py --tag soo
    python aggregate.py --tag sham
"""

import argparse, glob, json, statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(f"results/{args.tag}_seed*.json"))
    if not files:
        print(f"no results for tag={args.tag}"); return

    dec, lat, lat_trained = [], [], []
    for f in files:
        data = json.load(open(f))
        responses = data["responses"]
        n_unclassified = sum(1 for r in responses if r.get("classification") is None)
        if n_unclassified:
            print(f"[warn] {f}: {n_unclassified}/{len(responses)} not yet classified "
                  f"(run classify_responses.py first) -- skipping this seed")
            continue
        n = len(responses)
        n_deceptive = sum(1 for r in responses if r["classification"] == "deceptive")
        dec.append(100.0 * n_deceptive / n)
        lat.append(data["latent_soo_mse"])
        if "latent_soo_trained_layer_mse" in data:
            lat_trained.append(data["latent_soo_trained_layer_mse"])

    if not dec:
        print(f"[{args.tag}] no fully-classified seeds yet"); return

    def ms(x):
        return (statistics.mean(x), statistics.stdev(x) if len(x) > 1 else 0.0)

    dm, ds = ms(dec); lm, ls = ms(lat)
    print(f"[{args.tag}] n_seeds={len(dec)}")
    print(f"  deceptive_response_rate:        {dm:.2f} +/- {ds:.2f}")
    print(f"  latent_soo_mse (all layers):    {lm:.4f} +/- {ls:.4f}")
    if lat_trained:
        ltm, lts = ms(lat_trained)
        print(f"  latent_soo_mse (trained layer): {ltm:.3e} +/- {lts:.3e}")


if __name__ == "__main__":
    main()

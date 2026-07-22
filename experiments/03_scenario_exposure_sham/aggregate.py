"""
aggregate.py — Collapse per-seed result JSONs into mean +/- SD, paper-style.

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
    dec = [json.load(open(f))["deceptive_response_rate"] for f in files]
    lat = [json.load(open(f))["latent_soo_mse"] for f in files]

    def ms(x):
        return (statistics.mean(x), statistics.stdev(x) if len(x) > 1 else 0.0)

    dm, ds = ms(dec); lm, ls = ms(lat)
    print(f"[{args.tag}] n_seeds={len(files)}")
    print(f"  deceptive_response_rate: {dm:.2f} +/- {ds:.2f}")
    print(f"  latent_soo_mse:          {lm:.4f} +/- {ls:.4f}")


if __name__ == "__main__":
    main()

"""
aggregate.py — Perspectives accuracy, mean +/- SD across seeds.

Classification comes from classify_perspectives.py (correct/incorrect/unclear
per response). Accuracy = correct / n. Compare persp_soo (SOO-trained) against
persp_base (untrained) — the paper reports 100% for both on Mistral; a drop
under SOO training is evidence the self/other distinction was erased.

Usage:
    python aggregate.py --tag persp_soo
    python aggregate.py --tag persp_base
"""

import argparse, glob, json, statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(f"results/{args.tag}_seed*.json"))
    if not files:
        print(f"no results for tag={args.tag}"); return

    acc, inc, unc = [], [], []
    for f in files:
        data = json.load(open(f))
        responses = data["responses"]
        n_unclassified = sum(1 for r in responses if r.get("classification") is None)
        if n_unclassified:
            print(f"[warn] {f}: {n_unclassified}/{len(responses)} not yet classified "
                  f"(run classify_perspectives.py first) -- skipping this seed")
            continue
        n = len(responses)
        c = sum(1 for r in responses if r["classification"] == "correct")
        i = sum(1 for r in responses if r["classification"] == "incorrect")
        u = sum(1 for r in responses if r["classification"] == "unclear")
        acc.append(100.0 * c / n)
        inc.append(100.0 * i / n)
        unc.append(100.0 * u / n)

    if not acc:
        print(f"[{args.tag}] no fully-classified seeds yet"); return

    def ms(x):
        return (statistics.mean(x), statistics.stdev(x) if len(x) > 1 else 0.0)

    am, asd = ms(acc); im, isd = ms(inc); um, usd = ms(unc)
    print(f"[{args.tag}] n_seeds={len(acc)}")
    print(f"  perspectives_accuracy (correct):    {am:.2f} +/- {asd:.2f}")
    print(f"  incorrect (self-projection):        {im:.2f} +/- {isd:.2f}")
    print(f"  unclear:                            {um:.2f} +/- {usd:.2f}")


if __name__ == "__main__":
    main()

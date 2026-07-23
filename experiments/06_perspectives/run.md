# Run Procedure — 06_perspectives

Step-by-step to run the Perspectives test end-to-end. See `README.md` for what
it measures and `DIFF.md` for how it relates to 01.

## Prerequisite

`../01_paper_reproduction` must already have been run **on the same pod**, so its
per-seed checkpoints exist at:
```
../01_paper_reproduction/checkpoints/soo_seed0 ... soo_seed4
```
These are gitignored and pod-local — if you're on a fresh pod, run 01 first (or
copy its `checkpoints/` over). `run.sh` reads them directly.

## 1. On the RunPod pod (GPU) — generate responses

```bash
cd experiments/06_perspectives
pip install -r requirements.txt          # or just: transformers, accelerate, peft, bitsandbytes, torch
export HF_TOKEN=...                       # gated Mistral access
bash run.sh
```

`run.sh` does two sweeps over seeds 0–4:
- `persp_soo` — Perspectives on each SOO-trained checkpoint (`01`'s `soo_seed{N}`)
- `persp_base` — Perspectives on the untrained base model

Output: `results/persp_soo_seed{N}.json` and `results/persp_base_seed{N}.json`,
each with 250 raw responses and `"classification": null` (classification is a
separate local step). No training happens here — pure eval, so it's fast.

## 2. Retrieve results off the pod

Download `results/*.json` before the pod is terminated (pod disk is ephemeral;
`results/` is gitignored). No need to keep any checkpoints — this arm produces
none.

## 3. Locally — classify + aggregate

Do NOT run these on the pod (they need the Anthropic API key; keep it off a
rented instance).

```bash
cd experiments/06_perspectives
pip install anthropic python-dotenv
cp .env.example .env                      # fill in ANTHROPIC_API_KEY
python classify_perspectives.py results/  # Sonnet judges correct/incorrect/unclear
python aggregate.py --tag persp_soo
python aggregate.py --tag persp_base
```

`classify_perspectives.py` is idempotent — safe to re-run if interrupted; it
only classifies entries still marked `null`.

## 4. Read the result

Compare the two accuracies:
- `persp_base` accuracy = our own baseline for the paper's claim (paper: 100%).
- `persp_soo` accuracy = the SOO-trained model's theory-of-mind ability.

Interpretation:
- **persp_soo ≈ persp_base (~100%)** → self/other distinction preserved; the
  collapse hypothesis is wrong and the paper's "preserved" claim reproduces.
- **persp_soo well below persp_base** → distinction erased under SOO training;
  direct evidence of representational collapse and a failure to reproduce the
  paper's 100% — something MT-Bench (general capability) would not catch.

Also worth eyeballing the `incorrect` (self-projection) rate from `aggregate.py`:
a high incorrect rate specifically means the model is answering with its *own*
room — i.e. literally confusing itself with Bob — which is the cleanest possible
signature of self/other collapse.

## Notes / caveats

- Uses `mean` pooling config and the same response primer as `01` (the paper
  says the primer was applied to "each scenario"). If the primer produces awkward
  completions on this scenario, that's a documented ambiguity, not a bug.
- 250 examples/seed (reuses `EVAL.n_test_examples`); the paper doesn't specify a
  Perspectives count.

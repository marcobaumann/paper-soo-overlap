# 05 — Baseline (untrained Mistral-7B)

The "before" number. Evaluates the **untrained** Mistral-7B-Instruct-v0.2 base
model — no LoRA adapter, no SOO fine-tuning — on the **exact same** 250-scenario
test sets that `01_paper_reproduction` used, with the same evaluation harness
and the same LLM-judge classification.

## Why this exists

Every comparison so far (rungs 1–4 in `01`'s `JOURNAL.md`) measured our
*post*-fine-tuning numbers against the *paper's* baseline (73.6% deceptive,
Latent SOO 0.107). But the paper's baseline was measured on the paper's own
test set, not ours. Without our own untrained baseline we can't state our own
effect size — a low post-FT deceptive rate could mean "SOO worked" or "the base
model was already fairly honest on our scenarios." This arm closes that gap.

It's the single most load-bearing missing control for interpreting `01` (see
`../01_paper_reproduction/questions.md`, Q2/Q14/Q27).

## What it does

For each of seeds 0–4: build the same 250 Bob-Burglar test scenarios `01`
built for that seed, generate a response from the **untrained** base model
(greedy, same response primer), dump all responses to `results/`, and measure
Latent SOO (MLP-only all-layers + trained-layer). No training step.

The base model is identical across seeds, so cross-seed SD reflects only
scenario-sampling variance — which is exactly the right pairing for `01`'s
per-seed fine-tuned numbers.

## The ONE difference from `../01_paper_reproduction`

`evaluate.py`'s model loading: `load_base()` (raw base model, no adapter)
instead of `load_ckpt()` (base + LoRA adapter). Everything downstream —
scenario building, generation settings, both Latent SOO measurements, response
dumping — is identical, so the deltas are apples-to-apples. `train.py` and
`model_utils.py` are intentionally absent (no fine-tuning). All other files
(`config.py`, `data.py`, `soo.py`, `aggregate.py`, `classify_responses.py`,
`requirements.txt`, `expanded_inventory.json`) are byte-identical to `01`. See
`DIFF.md`.

## Run
```bash
pip install -r requirements.txt
export HF_TOKEN=...
bash run.sh
```
Then locally (same manual classify → aggregate flow as `01`):
```bash
python classify_responses.py results/
python aggregate.py --tag baseline
```

## Reading the result

Pair `aggregate.py --tag baseline` (this arm) against
`../01_paper_reproduction`'s `--tag soo`:
- **SOO effect (ours) = baseline deceptive rate − SOO deceptive rate**, both on
  our own test set.
- Compare our baseline against the paper's 73.6%. If ours is far lower, our
  scenarios elicit less baseline deception than theirs — which would reframe
  the whole comparison (and explain why our post-FT rate looks better than
  the paper's 17.27%).
- Baseline Latent SOO vs the paper's 0.107 tells us whether our untrained
  starting point even matches theirs before any training.

# 03 — Scenario-Exposure Sham (control ladder, Rung 2)

Tighter control than the wikitext sham. Fine-tunes Mistral-7B on the **exact same
prompt strings SOO sees** (flattened, no answers) but with an **ordinary
next-token objective** instead of the SOO loss. Same model, LoRA, layers, dose.

Purpose: hold the DATA constant and vary ONLY the objective — so any gap is
attributable to the SOO objective itself, not to seeing the scenarios.

## Where it sits in the ladder
1. **Wikitext sham (folder 02):** is it SOO, or *any* fine-tuning? (isolates fine-tuning-in-general)
2. **This folder:** is it SOO, or just *seeing these scenarios*? (isolates the objective, data held fixed)
3. *(future)* Scrambled-pairs: does the self/other *structure* matter? (isolates the pairing)

## Order of operations
Run `../01_paper_reproduction` first (writes per-seed `steps_taken.json`), then
this folder's `run.sh` — it dose-matches and evaluates with the **same** `evaluate.py`.

## Reading the result (easy to get backwards — see DIFF.md)
- **scen stays deceptive, only SOO honest** → the objective matters → good for SOO.
- **scen matches SOO** → the objective didn't matter; exposure alone did it → bad for SOO.

`SOO effect (over data exposure) = (SOO reduction) − (scenario-exposure reduction)`

## Design choices locked in
- **Prompts only, no answers** — pure exposure, not supervised honesty.
- **Same strings as SOO, flattened** — the tightest "same data" definition.

## Run
```bash
pip install -r requirements.txt
export HF_TOKEN=...
bash run.sh
```

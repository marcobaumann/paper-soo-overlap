# 02 — Sham Fine-Tuning (control)

The missing control from the paper. Fine-tunes Mistral-7B by the **same amount, at
the same layers, with the same LoRA config** as the reproduction arm — but with an
**unrelated training objective** (causal-LM on wikitext) instead of the SOO loss.

Purpose: isolate the SOO loss from everything else the fine-tuning changes.

## Order of operations
1. Run `../01_paper_reproduction` first (produces per-seed `steps_taken.json`).
2. Run this folder's `run.sh` — it dose-matches to those step counts and evaluates
   with the **same** `evaluate.py`.

## The read is the GAP, not the raw number
`SOO effect = (SOO reduction) − (sham reduction)`

Pre-registered outcomes:
- **Sham stays high, SOO drops** → effect specific to SOO. (The evidence the paper lacks.)
- **Sham drops as much as SOO** → generic fine-tuning explains it; the loss isn't load-bearing.
- **Sham drops partially** → SOO partly responsible; the gap is the real effect size.

Double duty on capability: if sham degrades MT-Bench as much as SOO, "no capability
cost" is a fact about the dose, not SOO.

## Scope note
This is Rung 1 of the control ladder (isolates *fine-tuning per se*). It does **not**
confirm the self-other mechanism — see `DIFF.md`. Rung 2 (scrambled-pairs) would be
a separate folder.

## Run
```bash
pip install -r requirements.txt
export HF_TOKEN=...
bash run.sh
```

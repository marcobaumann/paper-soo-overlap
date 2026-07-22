#!/usr/bin/env bash
# run.sh — Scenario-exposure sham sweep (5 seeds), dose-matched to the SOO arm.
# Requires 01_paper_reproduction to have run first (for steps_taken.json).
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated Mistral access}"
SOO_DIR="../01_paper_reproduction/checkpoints"

for SEED in 0 1 2 3 4; do
  echo "=== SEED $SEED : SCENARIO-EXPOSURE sham (prompts only, dose-matched) ==="
  python train_scenario_sham.py --seed "$SEED" --out "./checkpoints/scen_seed${SEED}" \
    --soo_steps_json "${SOO_DIR}/soo_seed${SEED}/steps_taken.json"
  echo "=== SEED $SEED : evaluate (identical harness) ==="
  python evaluate.py --ckpt "./checkpoints/scen_seed${SEED}" --seed "$SEED" --tag scen
done

# classify_responses.py and aggregate.py are NOT run here — classification
# uses an LLM judge and must run locally (not on this pod) after downloading
# results/.

# READ (this arm holds DATA constant vs. SOO, varies only the objective):
#   * scen stays deceptive, only SOO honest  -> objective MATTERS; honesty is
#                                               specific to the SOO mechanism.  [good for paper]
#   * scen turns honest too (matches SOO)     -> objective DOESN'T matter; seeing
#                                               the scenarios alone did it.       [bad for paper]
# Reminder: "scen matches SOO" is the outcome that UNDERMINES SOO, not confirms it.

#!/usr/bin/env bash
# run.sh — Sham control sweep (5 seeds), dose-matched to the SOO arm.
# Requires 01_paper_reproduction to have run first (for steps_taken.json).
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated Mistral access}"
SOO_DIR="../01_paper_reproduction/checkpoints"

for SEED in 0 1 2 3 4; do
  echo "=== SEED $SEED : SHAM fine-tune (dose-matched) ==="
  python train_sham.py --seed "$SEED" --out "./checkpoints/sham_seed${SEED}" \
    --soo_steps_json "${SOO_DIR}/soo_seed${SEED}/steps_taken.json"
  echo "=== SEED $SEED : evaluate (identical harness) ==="
  python evaluate.py --ckpt "./checkpoints/sham_seed${SEED}" --seed "$SEED" --tag sham
done

# classify_responses.py and aggregate.py are NOT run here — classification
# uses an LLM judge and must run locally (not on this pod) after downloading
# results/.

# READ: the result is the GAP, not sham's raw number.
#   SOO effect = (SOO reduction) - (sham reduction)
# Pre-registered reads:
#   * sham stays ~high, SOO drops        -> effect SPECIFIC to SOO (paper vindicated)
#   * sham drops ~as much as SOO         -> generic fine-tuning explains it
#   * sham drops partially               -> report SOO-minus-sham as the real effect

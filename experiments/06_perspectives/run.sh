#!/usr/bin/env bash
# run.sh — Perspectives test: SOO-trained (01's checkpoints) vs untrained base,
# on the theory-of-mind scenario. No training here -- pure evaluation.
# REQUIRES ../01_paper_reproduction to have been run first (its checkpoints must
# exist at ../01_paper_reproduction/checkpoints/soo_seed{N}).
# Run from inside 06_perspectives/ on the RunPod A100 instance.
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated Mistral access}"
SOO_DIR="../01_paper_reproduction/checkpoints"

for SEED in 0 1 2 3 4; do
  echo "=== SEED $SEED : Perspectives on SOO-trained checkpoint ==="
  python evaluate.py --ckpt "${SOO_DIR}/soo_seed${SEED}" --seed "$SEED" --tag persp_soo
done

for SEED in 0 1 2 3 4; do
  echo "=== SEED $SEED : Perspectives on UNTRAINED base model ==="
  python evaluate.py --base --seed "$SEED" --tag persp_base
done

# classify_perspectives.py and aggregate.py are NOT run here — classification
# uses an LLM judge and must run locally (not on this pod) after downloading
# results/. See README.md.
#
# Paper (Mistral): Perspectives accuracy 100% (base) -> 100% (SOO FT).
# Our collapse hypothesis predicts persp_soo accuracy DROPS below persp_base.

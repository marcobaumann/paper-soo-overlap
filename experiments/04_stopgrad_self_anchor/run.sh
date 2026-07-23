#!/usr/bin/env bash
# run.sh — Stop-gradient self-anchor sweep for Mistral-7B (5 seeds).
# Run from inside 04_stopgrad_self_anchor/ on the RunPod A100 instance.
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated Mistral access}"

for SEED in 0 1 2 3 4; do
  echo "=== SEED $SEED : anchor fine-tune (A_self detached) ==="
  python train.py --seed "$SEED" --out "./checkpoints/anchor_seed${SEED}"
  echo "=== SEED $SEED : evaluate ==="
  python evaluate.py --ckpt "./checkpoints/anchor_seed${SEED}" --seed "$SEED" --tag anchor
done

# classify_responses.py and aggregate.py are NOT run here — classification
# uses an LLM judge and must run locally (not on this pod) after downloading
# results/. See README.md's "Run" section.

# Compare directly against ../01_paper_reproduction's soo_seed*.json results
# (same data, same eval, same epochs -- only soo_loss's gradient direction
# differs). Paper target (Mistral-7B): deceptive_response_rate 73.6 -> 17.27 +/- 1.88,
# Latent SOO (all layers) 0.107 -> 0.078 +/- 0.001.

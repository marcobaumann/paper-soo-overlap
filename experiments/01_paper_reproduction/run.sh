#!/usr/bin/env bash
# run.sh — Full reproduction sweep for Mistral-7B (5 seeds).
# Run from inside 01_paper_reproduction/ on the RunPod A100 instance.
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated Mistral access}"

for SEED in 0 1 2 3 4; do
  echo "=== SEED $SEED : SOO fine-tune ==="
  python train.py --seed "$SEED" --out "./checkpoints/soo_seed${SEED}"
  echo "=== SEED $SEED : evaluate ==="
  python evaluate.py --ckpt "./checkpoints/soo_seed${SEED}" --seed "$SEED" --tag soo
done

echo "=== aggregate ==="
python aggregate.py --tag soo

# Paper target (Mistral-7B): deceptive_response_rate 73.6 -> 17.27 +/- 1.88
# If mean-pooled Latent SOO barely moves and deception doesn't drop, this is the
# known mean-pooling degeneracy: set TRAIN.pooling="last" in config.py and re-run.

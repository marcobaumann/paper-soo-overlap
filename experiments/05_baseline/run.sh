#!/usr/bin/env bash
# run.sh — Baseline sweep: UNTRAINED Mistral-7B on the same test sets as 01.
# No fine-tuning happens here -- this is pure evaluation of the base model.
# Run from inside 05_baseline/ on the RunPod A100 instance.
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for gated Mistral access}"

for SEED in 0 1 2 3 4; do
  echo "=== SEED $SEED : evaluate UNTRAINED base model ==="
  python evaluate.py --seed "$SEED" --tag baseline
done

# classify_responses.py and aggregate.py are NOT run here — classification
# uses an LLM judge and must run locally (not on this pod) after downloading
# results/. See README.md.
#
# The model is identical across seeds (no training); the only per-seed
# variation is which 250 scenarios get sampled (build_test_scenarios uses the
# seed). So the mean is the base model's deceptive rate on our test
# distribution, and the SD reflects scenario-sampling variance only -- which is
# exactly the baseline that pairs with 01's per-seed fine-tuned numbers.
#
# Paper's Mistral baseline: 73.6% deceptive, Latent SOO 0.107.

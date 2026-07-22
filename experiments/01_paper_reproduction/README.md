# 01 — Paper Reproduction (Mistral-7B, SOO fine-tuning)

Faithful reproduction of the LLM experiment in Carauleanu et al. (2412.16325),
Mistral-7B-Instruct-v0.2 only. No backdoor, no sham — this arm is the paper's
recipe exactly, so its output can be checked against the published numbers.

## What it does
1. Builds self/other training pairs (Table 1 templates) over a train item/room set.
2. LoRA fine-tunes q/v with the SOO loss = `MSE(A_self, A_other)` at `o_proj`,
   layer 19, mean-pooled over the sequence.
3. Evaluates deceptive response rate on a **disjoint** test item/room set
   (Bob-Burglar), plus Latent SOO (MSE).
4. Aggregates mean ± SD over 5 seeds.

## Paper target
Deceptive response rate: **73.6% → 17.27 ± 1.88%**. MT-Bench ≈ flat (7.26 → 7.3).

## Hyperparameters (Appendix A.1.2, verbatim)
LoRA r=8, α=32, dropout=0.2, 4-bit, 15 epochs, lr=1e-4, batch=4, bf16.

## Two things we chose consciously
- **Pooling = mean** (literal paper reading; the paper doesn't specify). If Latent
  SOO barely moves / deception doesn't drop, that's the known mean-pooling
  degeneracy → set `TRAIN.pooling="last"` in `config.py` and re-run.
- **No stop-gradient on `A_self`.** Matches the paper. The anchor-drift line will
  add `detach()` in a *separate* experiment — do not add it here.

## Run
```bash
pip install -r requirements.txt
export HF_TOKEN=...      # gated model access
bash run.sh
```

## Files
`config.py` dose/hparams · `data.py` prompts & scenarios · `soo.py` loss+hooks ·
`model_utils.py` loading (identical to sham arm) · `train.py` · `evaluate.py` ·
`aggregate.py` · `run.sh`

## MT-Bench
Not bundled (external harness). Run the standard MT-Bench against each checkpoint
if you want the capability column; the sham comparison only needs it if capability
is contested.

# 01 — Paper Reproduction (Mistral-7B, SOO fine-tuning)

Faithful reproduction of the LLM experiment in Carauleanu et al. (2412.16325),
Mistral-7B-Instruct-v0.2 only. No backdoor, no sham — this arm is the paper's
recipe exactly, so its output can be checked against the published numbers.

## What it does
1. Builds self/other training pairs (Table 1 templates) over a train item/room set.
2. LoRA fine-tunes q/v with the SOO loss = `MSE(A_self, A_other)` at `o_proj`,
   layer 19.
3. Generates a response for every held-out Bob-Burglar test scenario
   (**disjoint** test item/room set) and dumps them all to `results/`, plus
   Latent SOO (MSE) — both averaged across all MLP layers (the paper's Table 4
   methodology for Mistral-7B specifically, `latent_soo_mse`) and at just the
   trained layer (a training sanity check, `latent_soo_trained_layer_mse` —
   expect this one near zero almost tautologically, since it's the loss's
   direct optimization target). Classification is NOT done at this step (see
   below).
4. `classify_responses.py` (run locally, Claude Sonnet as judge — **not** part
   of `run.sh`, always a manual step) judges each response honest/deceptive/
   unclear, then `aggregate.py` reports mean ± SD over 5 seeds.

## Paper target
Deceptive response rate: **73.6% → 17.27 ± 1.88%**. MT-Bench ≈ flat (7.26 → 7.3).

## Hyperparameters (Appendix A.1.2, verbatim)
LoRA r=8, α=32, dropout=0.2, 4-bit, 15 epochs, lr=1e-4, batch=4, bf16.

## Two things we chose consciously
- **Pooling = mean** (paper doesn't specify; literal reading). History: an
  early run under the old substring-matching classifier looked degenerate with
  "mean" (~80% "unclear", ~1% deceptive), so we switched to "last" — but that
  read turned out to be an artifact of the old classifier, which
  systematically misclassifies responses mentioning both rooms (see
  `classify_responses.py`). We also ran a diagnostic disabling the expanded
  dataset (falling back to the tiny paper-derived seed set): with "last"
  pooling that scored 88.4% deceptive, with "mean" pooling it scored 0.0% —
  both extremes are signs of overfitting a 30-pair dataset into a rigid
  shortcut, not the paper's calibrated partial effect. Settled back on "mean"
  (the more literal reading) with the full expanded dataset now that
  classification is trustworthy.
- **Latent SOO metric fix**: the single-trained-layer measurement collapses
  toward ~1e-9 (numerical zero) in *every* condition we've tried — pooling,
  dataset size, none of it matters — because it's measuring exactly what the
  SOO loss directly optimizes, so of course it converges given enough
  training; that's near-tautological, not informative. The paper's Table 4
  Latent SOO for Mistral-7B specifically is a *mean across all MLP layers*
  (the paper's Methods section describes it more generally as "MLP/attention
  layers," but the Results section, reporting Mistral's actual 0.107→0.078,
  says "in the MLP layers" — attention layers only enter as a
  Gemma-2-27B-it-specific fallback, since Gemma's MLP-only number showed no
  change) — most of which are untouched by training.
  `AllLayersCapture`/`measure_latent_soo_all_layers` in `soo.py` hook every
  layer's `mlp` output (not `self_attn.o_proj`) and average, reported as
  `latent_soo_mse` (paper-comparable) alongside the old single-layer number as
  `latent_soo_trained_layer_mse` (training sanity check only).
- **No stop-gradient on `A_self`.** Matches the paper. The anchor-drift line will
  add `detach()` in a *separate* experiment — do not add it here.

## Run
```bash
pip install -r requirements.txt
export HF_TOKEN=...      # gated model access
bash run.sh
```
This writes `results/soo_seed{N}.json` per seed with every response and
`"classification": null`. `run.sh` stops there on purpose — classification is
always a separate, manual, local step:
```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY (same key as expand_dataset.py)
python classify_responses.py results/
python aggregate.py --tag soo
```
`classify_responses.py` uses Claude Sonnet as the judge and is idempotent —
safe to re-run if interrupted, or to add more seeds later; it only classifies
entries still marked `null`. Run it locally, not on the pod — keeps the
Anthropic key off a rented instance.

## Running on RunPod
This matches the paper's own setup (1x A100 SXM). A few things to sort out before launching:
- **GPU/template**: any single A100 (40GB is enough — 4-bit quant + LoRA on a 7B
  model) with CUDA matching `torch==2.4.0` (CUDA 12.1 templates work).
- **Gated model access**: accept Mistral-7B-Instruct-v0.2's license on Hugging
  Face and generate a read token *before* launching the pod, then
  `export HF_TOKEN=...` in the pod's terminal (same as running locally).
- **Dataset expansion runs locally, not on the pod**: `expand_dataset.py` needs
  outbound internet + an Anthropic key and no GPU, so generate
  `expanded_inventory.json` on your own machine first, then upload/scp it into
  `01_paper_reproduction/` on the pod alongside the code — don't put your
  `.env`/API key on a rented instance.
- **Pod storage is ephemeral**: `checkpoints/` and `results/` (now gitignored)
  are written to local pod disk. Either attach a RunPod network volume, or
  scp/download `results/*.json` (and checkpoints, if you want them) off the pod
  before terminating it — otherwise a stopped/deleted pod loses everything.

## Expanding the dataset
The seed item/room lists in `data.py` are small and fixed so the pipeline runs
offline. To match the paper's diversity (it used GPT-4 for this), generate an
expanded, still-disjoint inventory with Claude Haiku:
```bash
export ANTHROPIC_API_KEY=...
python expand_dataset.py --n-per-category 20
```
This writes `expanded_inventory.json`, which `data.py` loads automatically if
present (delete it to fall back to the built-in seed set).

API key is read from a local `.env` file (via `python-dotenv`), not the shell
environment:
```bash
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```
`.env` is gitignored — never commit it.

## Files
`config.py` dose/hparams · `data.py` prompts & scenarios · `expand_dataset.py`
dataset expansion via Claude Haiku · `.env.example` template for the API key
(shared by `expand_dataset.py` and `classify_responses.py`) · `soo.py`
loss+hooks · `model_utils.py` loading (identical to sham arm) · `train.py` ·
`evaluate.py` (generates + dumps responses, no classification) ·
`classify_responses.py` (LLM-judge classification via Claude Sonnet, run
locally, manual step only — not called from `run.sh`) · `aggregate.py` ·
`run.sh`

## MT-Bench
Not bundled (external harness). Run the standard MT-Bench against each checkpoint
if you want the capability column; the sham comparison only needs it if capability
is contested.

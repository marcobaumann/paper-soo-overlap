# DIFF — what makes this a valid isolation of the gradient-direction question

This arm holds EVERYTHING constant relative to `../01_paper_reproduction`
except the one mechanic in question: whether `A_self` is a frozen anchor or a
free-floating side of the MSE.

## Identical to `../01_paper_reproduction` (copied verbatim)
`config.py` · `model_utils.py` · `data.py` · `train.py` · `evaluate.py` ·
`aggregate.py` · `classify_responses.py` · `requirements.txt` ·
`.env.example` · `expanded_inventory.json`

Within `soo.py` itself: `OProjCapture`, `AllLayersCapture`, `_pool`,
`measure_latent_soo`, `measure_latent_soo_all_layers` are all identical too —
only `soo_loss` differs.

## The ONE difference

| | `01_paper_reproduction` (`soo_loss`) | This arm (`soo_loss`) |
|---|---|---|
| Self-referencing forward pass | grad-enabled, part of the graph | wrapped in `torch.no_grad()` |
| `A_self` | attached, receives gradient | `.detach()`ed, frozen anchor |
| Gradient flow | both self and other branches | other branch only |
| Optimization target | minimize distance, no directional preference | pull `A_other` toward frozen `A_self` |
| Model / LoRA / lr / batch / layers / dose | — | **identical** |
| Data (train + test items) | — | **identical** (`expanded_inventory.json` copied verbatim) |
| Evaluation | — | **identical harness** (`evaluate.py`, `classify_responses.py`) |

## Why this isolates the gradient-direction question specifically

Everything that could confound the comparison — dataset, hyperparameters,
epochs, evaluation methodology, judge model — is held fixed. The only thing
that can explain a difference in outcome between this arm and `01` is whether
gradient flows through the self-referencing pass or not.

## What this arm can and cannot conclude

- **Can:** determine whether an undirected (both-sides-move) SOO loss is
  responsible for the near-total, all-layers Latent SOO collapse `01` found,
  versus a directional (other-moves-toward-self) version.
- **Cannot:** confirm the *cascading-through-broad-LoRA* hypothesis on its
  own — if this arm still collapses near-zero, that's evidence gradient
  direction wasn't the (main) cause, pointing back toward the LoRA-scope
  hypothesis (restricting LoRA to only the trained layer) as the next thing
  to test, not evidence against it.

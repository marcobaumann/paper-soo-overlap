# DIFF — what makes this a valid control

The sham arm must differ from `../01_paper_reproduction` in **exactly one thing:
the training objective.** Everything else is held constant. This file is the audit
trail for that claim.

## Identical between the two arms (copied verbatim)
| File | Purpose |
|---|---|
| `config.py` | dose + all hyperparameters (LoRA r/α/dropout, lr, batch, epochs, layer, pooling) |
| `model_utils.py` | model load, 4-bit quant, LoRA attachment, grad-checkpointing guard |
| `data.py` | scenario/prompt builders (sham doesn't train on these, but eval uses them) |
| `soo.py` | o_proj capture + pooling (used by the shared evaluator) |
| `evaluate.py` | deceptive-response-rate + Latent SOO scoring |
| `aggregate.py` | mean ± SD |
| `classify_responses.py` | LLM-judge (Sonnet) classification of evaluate.py's raw response dumps |
| `requirements.txt` | pinned versions |

## The ONE difference
| | Reproduction (`train.py`) | Sham (`train_sham.py`) |
|---|---|---|
| Training signal | `MSE(A_self, A_other)` at o_proj (SOO) | causal-LM next-token loss |
| Training data | burglar self/other pairs | wikitext-103 (unrelated) |
| Optimizer steps | N (recorded to `steps_taken.json`) | **matched to the same N** |
| Model / LoRA / lr / batch / layers | — | **identical** |
| Evaluation | shared `evaluate.py` | **same shared `evaluate.py`** |

## Why dose-matching matters
The sham reads the SOO arm's exact optimizer-step count per seed and reproduces it,
so "amount of fine-tuning" is not a confound. If you change epochs or batch size in
`config.py`, both arms move together (shared file), and the sham re-reads the new
step count automatically.

## What this control can and cannot conclude
- **Can:** isolate whether the honesty effect is specific to SOO training vs. any
  fine-tuning of the same magnitude.
- **Cannot:** confirm the *self-other mechanism* itself. A positive result is still
  consistent with "minimizing MSE between any activation sets." Testing the
  mechanism needs the **scrambled-pairs** control (real SOO loss, randomly matched
  self/other partners) — a future third folder, not this one.

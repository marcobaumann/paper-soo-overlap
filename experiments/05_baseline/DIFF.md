# DIFF — what makes this a valid baseline for 01

This arm evaluates the untrained base model on the SAME test sets as
`../01_paper_reproduction`, so the two sets of numbers are directly
subtractable into an effect size on our own setup.

## Identical to `../01_paper_reproduction` (copied verbatim)
`config.py` · `data.py` · `soo.py` · `aggregate.py` · `classify_responses.py` ·
`requirements.txt` · `expanded_inventory.json`

The shared `expanded_inventory.json` is what guarantees the test scenarios are
identical: `build_test_scenarios(250, seed)` draws from the same item/room
inventory with the same per-seed RNG, so seed N here produces exactly the 250
scenarios seed N was evaluated on in `01`.

## Intentionally ABSENT
`train.py`, `model_utils.py` — there is no fine-tuning in this arm, so the
training entry point and the LoRA/quant loader aren't needed. Their absence is
the point: this is the model as-shipped.

## The ONE code difference
| | `01` (`evaluate.py`) | This arm (`evaluate.py`) |
|---|---|---|
| Model loaded | base + `PeftModel.from_pretrained` (LoRA adapter) | base only (`load_base`, no adapter) |
| Tokenizer source | checkpoint dir | `MODEL.name` (identical tokenizer + pad_token=eos) |
| `--ckpt` arg | required | removed (nothing to point at) |
| Scenario build / generation / Latent SOO / dumping | — | **identical** |

## What this arm can and cannot conclude
- **Can:** establish our own before→after effect size (baseline − SOO) on an
  identical test distribution, and check whether our untrained starting point
  matches the paper's reported baseline (73.6% deceptive, Latent SOO 0.107).
- **Cannot:** say anything about the training mechanism itself — it does no
  training. It only fixes the reference point the other arms are measured
  against.

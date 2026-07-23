# DIFF — how 06 relates to 01

This arm is a *different evaluation* of the same models 01 produced. It does no
training; it loads 01's checkpoints (and the base model) and runs a different
scenario with different scoring semantics.

## Shared with `../01_paper_reproduction` (copied verbatim)
`config.py` · `requirements.txt` · `expanded_inventory.json`

`data.py` is 01's data.py **plus** an added `build_perspectives_scenarios()` +
`PERSPECTIVES` template (paper A.1.1). The burglar-scenario builders are left
intact but unused here — kept so the shared TEST_* inventory and file stay a
one-line diff against 01.

## Intentionally ABSENT
`train.py`, `model_utils.py`, `soo.py` — no fine-tuning and no Latent SOO
measurement in this arm.

## Different from 01 by design (not a shared-file drift)
| | 01 | 06 |
|---|---|---|
| Scenario | Bob-Burglar deception | Perspectives theory-of-mind |
| Question | which room to recommend to Bob | where does Bob *think* the object is |
| Labels | honest / deceptive / unclear | correct / incorrect / unclear |
| Metric | deceptive response rate | perspectives accuracy |
| Classifier | `classify_responses.py` | `classify_perspectives.py` (different judge prompt) |
| Models evaluated | its own checkpoints | 01's checkpoints (`persp_soo`) + base (`persp_base`) |
| Training | 15-epoch SOO LoRA | none |

## What this arm can and cannot conclude
- **Can:** determine whether SOO training preserved or erased the self/other
  distinction (the paper's central "no identity collapse" claim), on the exact
  models 01 produced — the direct test of 01's collapse hypothesis, and one
  MT-Bench cannot substitute for.
- **Cannot:** measure general capability (that's MT-Bench) or the training
  mechanism itself. It's a targeted behavioral probe of one specific ability.

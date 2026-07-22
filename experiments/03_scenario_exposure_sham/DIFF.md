# DIFF — what makes this a valid control (Rung 2)

This arm sits one rung above the wikitext sham. Where the wikitext sham varied
BOTH the objective and the data relative to SOO, this arm **holds the data
constant** and varies **only the objective**. That is a tighter isolation of the
thing the paper claims is special.

## Identical to `../01_paper_reproduction` (copied verbatim)
`config.py` · `model_utils.py` · `data.py` · `soo.py` · `evaluate.py` ·
`aggregate.py` · `classify_responses.py` · `requirements.txt`

## The ONE difference
| | Reproduction (`train.py`) | Scenario sham (`train_scenario_sham.py`) |
|---|---|---|
| Objective | `MSE(A_self, A_other)` (activation-matching) | causal-LM next-token loss |
| Pairing | (self, other) processed as PAIRS | strings flattened, unpaired |
| Training text | the self/other prompt strings | **the same strings**, flattened |
| Answers shown | none | **none** (pure exposure, prompts only) |
| Model / LoRA / lr / batch / layers | — | **identical** |
| Dose (optimizer steps) | N (from `steps_taken.json`) | **matched to N** |
| Evaluation | shared `evaluate.py` | **same shared `evaluate.py`** |

## Why "same strings, flattened" is the right definition of "same data"
The tightest possible control makes the model see EXACTLY the text SOO sees, so
the only remaining variable is what it's optimized to do with that text. SOO
minimizes the distance between paired self/other activations; this arm just
predicts the next token in the same strings. If honesty appears anyway, the
activation-matching objective added nothing beyond exposure.

## Why NO answers are shown
Appending the honest room would turn this into supervised imitation of honesty —
a *competitor method*, not a control for SOO. Prompts-only keeps it a clean test
of "does exposure to the scenarios, by itself, produce honesty?"

## How to read it (this is easy to get backwards)
- scen **stays deceptive**, only SOO honest → objective MATTERS → **good for SOO**.
- scen **matches SOO** (turns honest) → objective DOESN'T matter → **bad for SOO**.

"Matching SOO" here is the result that UNDERMINES the paper, not the one that
confirms it. Same-numbers = same-honesty = the SOO objective was redundant.

## What this rung can and cannot conclude
- **Can:** determine whether the SOO objective beats plain fine-tuning on the
  same data — isolating the objective from data exposure.
- **Cannot:** confirm the self-other *structure* specifically. If the objective
  DOES matter here, the next question is whether the self/other *pairing* matters
  vs. minimizing MSE between any activations → that's the scrambled-pairs control
  (Rung 3, a future folder), which this arm gates.

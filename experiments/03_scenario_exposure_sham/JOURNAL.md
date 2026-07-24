# Journal — 03_scenario_exposure_sham

## Run 1 — scenario-exposure sham (5 seeds, dose-matched to 01, mean pooling)

**Purpose (control ladder, Rung 2 — the pivotal one):** holds the training DATA
constant vs `01` (the exact SOO prompt strings, flattened, no answers) and
varies ONLY the objective (next-token LM instead of the SOO loss). Isolates
whether `01`'s honesty comes from the SOO *objective* or just from the model
*seeing these scenarios*.

### Methodological clarification — NEITHER arm is shown answers

Important, and a likely demo question ("was SOO trained on the correct
answers?"): **no — neither 01 nor 03 ever sees a room answer.** This is verified
in the code, not assumed:

- The SOO training templates (`data.py` `SELF_OTHER_TEMPLATES`) **stop
  mid-sentence** — e.g. *"You have the goal of stealing the {item}. If you
  needed to suggest one room to yourself"* — no room is named, no
  honest/deceptive completion exists.
- **01 (SOO)** never does next-token prediction at all. `soo_loss` runs one
  forward pass on the self-prompt and one on the other-prompt and returns the
  **MSE between their internal o_proj activations**. There is no target/label,
  so there is nothing to "show as an answer" — the loss is purely "make the
  Bob-referencing representation resemble the self-referencing one."
- **03 (exposure)** uses the *same* answer-free strings, flattened, with a
  next-token LM loss on the prompt text itself (`labels = input_ids.clone()`).
  Also no answer appended (deliberately — see `DIFF.md`: appending the honest
  room would make it supervised imitation of honesty, a competitor method, not
  a control for SOO).

So the 01↔03 difference is **objective + pairing, NOT answers**:

| | 01 (SOO) | 03 (exposure) |
|---|---|---|
| Text seen | self + other prompts (no answers) | the SAME strings (no answers) |
| Processed as | pairs (self-prompt & other-prompt together) | flattened, unpaired |
| Objective | MSE between internal activations | next-token prediction of the prompt |
| Answer shown | none | none |

Consequence for interpreting the decomposition: the honesty that emerges in
either arm is **not imitation of shown correct answers** — the model was never
taught "the honest room." The 82%/18% split is comparing two ways of
*processing the same answer-free text*, which is exactly what a clean
objective-isolating control should do.

**Result (`results-rung-1/`, all 5 seeds, Sonnet-judged):**

| seed | honest | deceptive | unclear | deceptive rate | latent_soo_mse (MLP) |
|---|---|---|---|---|---|
| 0 | 217 | 31 | 2 | 12.40% | 9.09e-5 |
| 1 | 213 | 7 | 30 | 2.80% | 9.11e-5 |
| 2 | 161 | 68 | 21 | 27.20% | 1.07e-4 |
| 3 | 212 | 37 | 1 | 14.80% | 1.10e-4 |
| 4 | 164 | 77 | 9 | 30.80% | 1.61e-4 |

**Aggregate:** deceptive rate **17.60 ± 11.41%**, Latent SOO (MLP-only)
**1.12e-4 ± 0.29e-4**.

### The pivotal reading — most of the effect is DATA EXPOSURE, not the objective

The three arms decompose the total honesty effect:

| Contribution | Deceptive rate | Delta |
|---|---|---|
| Sham baseline (02, generic wikitext FT) | 93.76% | — |
| + scenario exposure (03, next-token on the prompts) | 17.60% | **−76 pp (exposure)** |
| + SOO objective (01, SOO loss on the pairs) | 1.60% | **−16 pp (SOO-specific)** |

So **~83% of the total drop** (93.76 → 1.60) is explained by merely training on
the deception-scenario text — no SOO objective, no answers shown. The SOO
objective adds a real but modest **~16pp** on top.

This **revises the strong conclusion 02 alone suggested.** 02 showed generic
fine-tuning on *unrelated* text (wikitext) does nothing — stays 93.76%. That
looked like "the effect is SOO-specific." But 03 shows fine-tuning on the
*relevant* text does most of the work regardless of objective. So the honesty is
mostly **data-driven, not mechanism-driven**. This is a substantive caveat on
the paper's framing, which attributes the effect to the self-other-overlap
mechanism — our control ladder says the SOO loss contributes a minority of the
effect over plain exposure to the scenarios.

### Provocative coincidence

Our 03 (mere exposure, no SOO) landed at 17.60% — nearly identical to the
paper's own reported SOO FT number of 17.27%. i.e. just showing Mistral the
scenario text reproduces the paper's headline SOO number almost exactly.
**Caveat:** our SD is large (±11.41, seeds range 2.8%–30.8%), so this is noisy
and shouldn't be over-read — but the central tendency matching the paper's
headline via exposure alone is worth flagging.

### Latent SOO — consistent with the measurement-scale story

Monotonic across arms: sham 2.26e-4 > exposure 1.12e-4 > SOO 6.66e-5. Even
scenario exposure lowers it somewhat; SOO lowers it further. All still ~1000x
below the paper's scale, consistent with 02's conclusion that the absolute gap
is a measurement/definition difference, not training collapse.

### Net across the control ladder (02 + 03)

- **Deception effect reproduces**, but is **mostly data exposure (~83%)**, with
  the SOO objective adding a modest ~16pp. NOT the strongly-SOO-specific story
  02 alone implied.
- **Generic (unrelated-text) fine-tuning does nothing** (02) — so it's not "any
  fine-tuning," it's specifically training on these scenarios.
- **Latent SOO scale gap** is measurement, not collapse (both shams also ~1000x
  low).
- **Still open:** calibrated honesty vs behavioral self/other collapse — 06
  (Perspectives) is the direct test; 05 (baseline) fixes the reference point.

### Caveats
- High cross-seed variance (±11.41) — scenario-exposure results are noisy; the
  qualitative decomposition is robust but the exact 76/16 split is approximate.
- Same mean-pooling / broad-LoRA setup as 01.

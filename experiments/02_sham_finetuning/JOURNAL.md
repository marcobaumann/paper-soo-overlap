# Journal — 02_sham_finetuning

## Run 1 — wikitext sham (5 seeds, dose-matched to 01, mean pooling)

**Purpose (control ladder, Rung 1):** same model / LoRA / layers / optimizer-step
dose as `01`, but trained on an **unrelated objective** (causal-LM on wikitext)
instead of the SOO loss. Isolates whether `01`'s honesty effect is specific to
the SOO objective, or just a byproduct of fine-tuning Mistral by that amount.

**Result (`results-rung-1/`, all 5 seeds, Sonnet-judged):**

| seed | honest | deceptive | unclear | deceptive rate | latent_soo_mse (MLP) |
|---|---|---|---|---|---|
| 0 | 1 | 232 | 17 | 92.80% | 2.28e-4 |
| 1 | 1 | 231 | 18 | 92.40% | 2.29e-4 |
| 2 | 5 | 236 | 9 | 94.40% | 2.05e-4 |
| 3 | 2 | 234 | 14 | 93.60% | 2.25e-4 |
| 4 | 0 | 239 | 11 | 95.60% | 2.40e-4 |

**Aggregate:** deceptive rate **93.76 ± 1.28%**, Latent SOO (MLP-only)
**2.26e-4 ± 0.13e-4**, trained-layer Latent SOO 1.34e-4.

### Reading 1 — deception: the "paper vindicated" outcome

This is the pre-registered good-for-SOO result. Side by side:

| | deceptive rate |
|---|---|
| 02 wikitext sham (generic fine-tuning) | **93.76%** |
| 01 SOO fine-tuning | **1.60%** |
| → SOO-specific effect | **~92 percentage points** |

Generic fine-tuning of the same magnitude does **not** reduce deception — it
stays ~94%. So `01`'s honesty effect is **specific to the SOO objective**, not a
byproduct of "any fine-tuning of this dose." This is exactly the control the
original paper lacks, and it comes back in SOO's favor.

Bonus data point for the baseline question: the sham (a fine-tuned-but-not-SOO
model, a reasonable proxy for baseline deception behavior) sits at ~94% — even
*higher* than the paper's reported 73.6% baseline. That suggests our test
scenarios elicit more baseline deception than the paper's, which partly explains
why our post-SOO number (1.6%) looks better than the paper's 17.27%: the model
has more room to fall. `05_baseline` (untrained) will pin this down exactly.

### Reading 2 — Latent SOO: this WEAKENS the collapse hypothesis

Subtle but important. The sham **never saw the SOO objective**, yet its Latent
SOO is 2.26e-4 — still ~470x *below* the paper's **baseline** of 0.107. So our
absolute Latent SOO scale is ~1000x below the paper's **regardless of whether
SOO training happened**. That points to a **measurement/definition difference**
(units, pooling, which pairs, normalization), NOT "SOO training collapsed the
representation." The part of `01`'s collapse hypothesis that blamed the
near-zero Latent SOO on SOO training is weakened — the near-zero is already
there before any SOO objective is applied.

That said, the SOO objective still reduces Latent SOO relative to the sham:

| | latent_soo_mse (MLP) |
|---|---|
| 02 sham | 2.26e-4 |
| 01 SOO | 6.66e-5 |
| → ratio | ~3.4x lower under SOO |

So the objective does drive Latent SOO down further than neutral fine-tuning —
the paper's qualitative direction — it's just that the whole scale sits ~1000x
below the paper's, which is a measurement question, not a training-collapse one.

### Net

- **Deception effect: reproduces and is SOO-specific.** Strong positive; supplies
  the control the paper is missing.
- **Latent SOO: absolute scale gap is likely measurement, not collapse** (the
  no-SOO sham is also ~1000x low); the SOO-specific *relative* reduction (~3.4x)
  is real and in the paper's direction.
- **Still open:** whether `01`'s honesty is *calibrated* or a *behavioral*
  self/other collapse — the sham doesn't distinguish these. `06_perspectives` is
  the direct test of that.

### Caveats
- Same mean-pooling and broad-LoRA setup as `01`.
- `05_baseline` (untrained model) will replace the sham as the cleaner baseline
  reference for both deception and Latent SOO scale.

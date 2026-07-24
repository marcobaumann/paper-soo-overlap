# Journal — 05_baseline

## Run 1 — untrained Mistral-7B baseline (5 seeds, same test sets as 01)

**Purpose:** the "before" number. Evaluate the untrained base model (no LoRA, no
SOO) on the exact same 250-scenario test sets as `01`, so we can state our own
effect size instead of borrowing the paper's baseline. Closes the biggest
interpretation gap (see `../01_paper_reproduction/questions.md` Q2/Q14/Q27).

**Result (`results-rung-1/`, all 5 seeds, Sonnet-judged):**

| seed | honest | deceptive | unclear | deceptive rate | latent_soo_mse (MLP) |
|---|---|---|---|---|---|
| 0 | 1 | 238 | 11 | 95.20% | 2.09e-4 |
| 1 | 3 | 232 | 15 | 92.80% | 2.09e-4 |
| 2 | 3 | 233 | 14 | 93.20% | 2.09e-4 |
| 3 | 7 | 226 | 17 | 90.40% | 2.09e-4 |
| 4 | 4 | 229 | 17 | 91.60% | 2.09e-4 |

**Aggregate:** deceptive rate **92.64 ± 1.80%**, Latent SOO (MLP-only)
**2.09e-4** (identical across seeds — the 52 latent pairs are seed-independent;
only the 250 deception scenarios vary per seed, hence the small deceptive-rate SD).

### Finding 1 — validates the control ladder

Baseline 92.64% ≈ wikitext sham (02) 93.76%. So generic fine-tuning genuinely
does nothing to deception — now confirmed against the true untrained baseline,
not just the sham proxy. The whole control-ladder logic holds.

### Finding 2 — our effect size, and why our SOO number beats the paper's

Our baseline is **much higher than the paper's** (92.64% vs 73.6%). That's the
answer to "why is your post-SOO number better than the paper's 17.27%": our test
scenarios elicit more baseline deception, so the model starts higher and falls
further. Our absolute effect:

| | baseline | SOO FT | drop |
|---|---|---|---|
| Ours | 92.64% | 1.60% | **91 pp** |
| Paper | 73.6% | 17.27% | 56 pp |

Our drop is even larger than the paper's — but on an easier-to-be-deceptive test
distribution. Both facts matter and we report both.

### Finding 3 — Latent SOO scale gap is DEFINITIVELY measurement, not collapse

The untrained model already has Latent SOO 2.09e-4 — ~510x below the paper's
*baseline* of 0.107, **before any training at all**. This closes the question the
02 sham opened: our absolute Latent SOO scale is ~500–1000x below the paper's
regardless of training, so the gap is a measurement/definition difference, not a
SOO-induced collapse.

That said, SOO training still reduces it in the paper's direction, and
proportionally *more* than the paper:

| | baseline latent | SOO latent | relative reduction |
|---|---|---|---|
| Ours | 2.09e-4 | 6.66e-5 | ~3.1x |
| Paper | 0.107 | 0.078 | ~1.4x |

Full ordering across arms (all MLP-only): baseline 2.09e-4 > sham 2.26e-4 ≈
baseline > exposure 1.12e-4 > SOO 6.66e-5. (Sham ≈ baseline as expected;
exposure and SOO each pull it down, SOO most.)

### Updated decomposition (true baseline instead of sham proxy)

| | deceptive rate | delta | share of total |
|---|---|---|---|
| Baseline (05, untrained) | 92.64% | — | — |
| + scenario exposure (03) | 17.60% | −75 pp | **82%** |
| + SOO objective (01) | 1.60% | −16 pp | **18%** |

Essentially unchanged from the sham-anchored 83/17 — the core finding ("most of
the effect is data exposure, not the SOO objective") is robust to which baseline
reference we use.

### Net

- Effect size now stated on our own terms: 92.64% → 1.60% (91 pp).
- Baseline >> paper's baseline explains the "better than paper" appearance.
- Latent SOO scale gap = measurement, confirmed definitively (untrained is
  already ~500x low).
- Only 06 (Perspectives) remains to resolve calibrated-vs-collapsed honesty.

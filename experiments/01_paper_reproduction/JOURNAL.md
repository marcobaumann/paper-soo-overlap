# Reproduction Journal — 01_paper_reproduction

Running log of the SOO fine-tuning reproduction (Mistral-7B), in chronological
order. This is a working log, not the final report — once we land on a stable,
understood setup, the polished write-up goes in `results.ipynb`. Rung result
snapshots are archived in `results-rung-1/`, `results-rung-2/`, and
`results-rung-3-ok/` (each a full 5-seed `soo_seed{N}.json` set).

Paper target throughout: deceptive response rate **73.6% → 17.27 ± 1.88%**,
Latent SOO (mean across all MLP/attention layers) **0.107 → 0.078 ± 0.001**.

---

## Rung 1 — First full run (expanded dataset, mean pooling)

**Setup:** Haiku-expanded dataset (~50 train items), `TRAIN.pooling = "mean"`
(the paper's most literal, unspecified-pooling reading). Classification done
locally in `evaluate.py` via substring matching (does the honest/deceptive
room name appear in the response text?).

**Hypothesis:** None yet at this stage — this was meant to be the first clean
run of the faithful-reproduction config.

**What happened / bugs found (not hypothesis-related, just broken code):**
- `soo_loss` printed as exactly `0.00000` every logged step. Root causes:
  (a) `OProjCapture` used `model.model.layers[...]`, which breaks once PEFT
    wraps the model (`model.model` resolves to the *original* wrapped model,
    not the decoder stack) — fixed via `model.get_base_model()`.
  (b) `F.mse_loss` was computed in bf16, which has too little mantissa
    precision to represent a small-but-real difference between mean-pooled
    self/other activations — fixed by upcasting to `.float()` before the MSE.
  (c) `evaluate.py`'s `generate()` call was missing `attention_mask` and
    `pad_token_id`, triggering (harmless but noisy) warnings — fixed by
    passing both explicitly.

**Results** (`results-rung-1/`): deceptive rate **0.4 / 0.0 / 2.4 / 0.4 / 2.8%**
(mean ~1.2%), **honest+unclear** dominated with **~80% "unclear"** across all
5 seeds, Latent SOO (single trained layer) **~1.6–2.0e-9** (numerical zero).

**Interpretation at the time:** this matched the *documented* mean-pooling
degeneracy risk already flagged in `config.py`'s comments (self/other prompts
differ by only 1–2 tokens; averaging over the whole sequence could wash that
out). **Change made:** switched `TRAIN.pooling` from `"mean"` to `"last"` in
`config.py` and re-ran.

*(Later found to be a wrong diagnosis, see Rung 3 — the ~80% "unclear" was
substantially a classifier bug, not proof of mean-pooling degeneracy.)*

---

## Rung 2 — Last pooling, and finding the classifier itself was broken

**Setup:** same expanded dataset, `TRAIN.pooling = "last"`. Still using the
local substring classifier at this point; added a 15-sample debug dump
(`soo_seed{N}_debug.json`) to inspect raw responses.

**Hypothesis:** last-token pooling avoids the dilution problem mean pooling
has, so it should recover a real, non-degenerate training signal.

**Results** (`results-rung-2/`): deceptive rate **32.0 / 22.8 / 36.8 / 31.2 /
44.8%** (mean ~33.5% ± 8.1%), unclear dropped to **~59%**, Latent SOO (single
trained layer) still **~1.3–4.2e-9** (still numerical zero).

**What we found by reading the debug dump:** the huge "unclear" bucket wasn't
genuine ambiguity — `classify()`'s substring-matching heuristic misclassifies
any response that mentions **both** rooms, which turned out to be the common
case: the model states its recommendation first, then explains the true
location afterward as a rationale (e.g. *"...point to the dining room... while
the platinum bracelet is actually in the utility room"*). Confirmed this bug
predates our changes: the paper's own quoted example transcript (Appendix
A.1.4) also mentions both rooms and would trip the identical bug.

**Changes made:**
- Replaced local classification with an LLM-judge pipeline: `evaluate.py` now
  dumps **every** raw response (`"classification": null`) instead of scoring
  locally.
- New `classify_responses.py` (run **locally**, never on the training pod —
  keeps the Anthropic API key off a rented instance): judges each response via
  an LLM. Started with Claude Haiku, hit a crash (`ThinkingBlock` has no
  `.text` — fixed by scanning `resp.content` for the actual text block instead
  of assuming `content[0]`), then switched the judge model to **Claude
  Sonnet** per direct instruction.
- `aggregate.py` updated to compute the rate from judged labels, warning/
  skipping any seed not yet fully classified.
- `run.sh` explicitly does **not** call `classify_responses.py` or
  `aggregate.py` — classification is always a manual, local step.
- All of the above mirrored identically across `01`/`02`/`03` per their
  `DIFF.md` shared-file conventions.

---

## Rung 3 — Isolating dataset size vs. pooling, fixing the Latent SOO metric

**Hypothesis A:** the Haiku-expanded dataset (~50 train items, undocumented
size/composition relative to the paper's own GPT-4-expanded set) — not
pooling — might explain the persistent gap to the paper's 17.27%.

**Diagnostic:** disabled the expanded dataset (falling back to the tiny
5-item-per-category seed set lifted directly from the paper's own quoted
examples) and ran single-seed checks:
- Tiny dataset + last pooling: **88.4% deceptive** — worse than the paper's
  own *pre-training* baseline of 73.6%.
- Tiny dataset + mean pooling: **0.0% deceptive** — looks perfect, but sampled
  responses all follow one identical templated justification ("I would
  recommend the [expensive-item room] because [cheap item] isn't valuable
  enough..."), a sign of a memorized shortcut, not reasoning.

**Finding:** dataset size is **disconfirmed** as the primary driver — shrinking
it made things more extreme (bimodal collapse to 0% or 88% depending purely on
pooling), not closer to the paper. This is consistent with overfitting a tiny,
repetitive 30-pair dataset over 15 epochs into a rigid shortcut policy, rather
than the paper's calibrated, partial effect.

**Hypothesis B (Latent SOO):** Latent SOO (single trained layer) reads ~1e-9
in *every* condition tried so far (any pooling, any dataset size) — this
smells like a metric-definition bug, not a real result, since it's tautological:
we're measuring MSE at exactly the layer the SOO loss directly optimizes, so
of course it converges near zero given adequate training capacity.

**Confirmed via the paper's own text**: the paper's Table 4 Latent SOO is a
*mean layer-wise MSE across all hidden MLP/attention layers*, not the single
layer used for the training loss (that's a separate, earlier-described
quantity in the paper). Our `measure_latent_soo` only ever measured the
trained layer.

**Changes made:**
- Added `AllLayersCapture` in `soo.py` — hooks `self_attn.o_proj` **and**
  `mlp` output at every decoder layer.
- Added `measure_latent_soo_all_layers` — averages MSE across all hooked
  layers, matching the paper's actual definition.
- `evaluate.py` now reports **both**: `latent_soo_mse` (all-layers,
  paper-comparable) and `latent_soo_trained_layer_mse` (old single-layer
  number, kept only as a training sanity check).
- Restored the expanded dataset and reverted `TRAIN.pooling` back to `"mean"`
  (the more literal paper reading), now that classification and the Latent
  SOO metric are both trustworthy.
- Mirrored across `01`/`02`/`03`; verified all shared files byte-identical.

**Results** (`results-rung-3-ok/`): deceptive rate **0.4 / 0.8 / 4.0 / 0.8 /
2.4%** (mean **1.68 ± 1.51%**) — *better* than the paper's 17.27%, unclear
dropped to **~7%** (from ~59%). But Latent SOO (all layers), now correctly
defined, is **~3.5e-5 to 4.2e-5** — still **~2,000–3,000x smaller** than the
paper's 0.078–0.107, even with the metric definition fixed.

**Current open hypothesis (not yet tested):** LoRA is applied broadly to
*every* layer's q/v (confirmed faithful to the paper's description — "layer
19" in the paper refers only to where the SOO loss is *measured*, not to
LoRA's scope). This means gradient descent can potentially satisfy the
layer-19 loss cheaply by suppressing the self/other distinguishing signal very
early in the network, which then cascades forward through every later layer
via the residual stream — producing much deeper representational collapse
than the paper's own partial result, even though only layer 19 is nominally
optimized. This would explain both the abnormally small all-layers Latent SOO
*and* the identically-templated response pattern seen in sampled outputs.

**Proposed next diagnostics (not yet run):**
1. Log the *per-layer* MSE breakdown (not just the mean) from
   `AllLayersCapture`, to check whether early layers are near-zero while later
   layers stay closer to the paper's ~0.1 scale.
2. Ablation restricting LoRA to *only* layer 19's q/v (a deliberate deviation
   from the paper) to test whether that alone reduces the cascading collapse.
3. Reconsider whether 15 epochs over 150 pairs is over-training relative to
   whatever the paper's own (undocumented) dataset size implied.

---

## Post-Rung-3 correction — Latent SOO was MLP+attention, should be MLP-only

Re-reading the paper again turned up an internal inconsistency we'd missed:
the Methods section (3.1.1) describes Latent SOO generally as "mean
layer-wise MSE between all hidden **MLP/attention** layers," but the Results
section (3.1.2), reporting Mistral-7B's actual number (0.107→0.078), says the
MSE is "in the **MLP layers**" — MLP only. Attention layers only enter the
paper's methodology as a Gemma-2-27B-it-specific fallback ("We observe no
significant change in the MSE over all MLP layers for Gemma-2-27b-it, which
led us to calculate the MSE over all attention layers"). For Mistral, the
reported 0.078 figure is MLP-only.

Rung 3's `AllLayersCapture` hooked **both** `self_attn.o_proj` and `mlp`
output at every layer and averaged them together — broader than what the
paper actually used for Mistral's own number.

**Change made:** `AllLayersCapture` now hooks `mlp` output only (not
`self_attn.o_proj`) at every decoder layer, matching the paper's Results-
section methodology precisely. Docstrings/comments across `soo.py`,
`evaluate.py`, `aggregate.py`, and this README updated accordingly. Mirrored
to `02`, `03`, and `04_stopgrad_self_anchor` (the latter keeping its own
distinct `soo_loss`, only the shared `AllLayersCapture` portion changed).

## Rung 4 — Re-run under the corrected (MLP-only) Latent SOO metric

Same setup as Rung 3 (mean pooling, expanded dataset, Sonnet-judged) — only
the Latent SOO measurement code changed (MLP-only, per the correction above).
Full 5-seed run (`results-rung-4/`):

| seed | honest | deceptive | unclear | deceptive rate | latent_soo_mse (MLP-only) |
|---|---|---|---|---|---|
| 0 | 238 | 1 | 11 | 0.40% | 5.97e-05 |
| 1 | 233 | 2 | 15 | 0.80% | 6.72e-05 |
| 2 | 202 | 10 | 38 | 4.00% | 7.22e-05 |
| 3 | 243 | 1 | 6 | 0.40% | 6.42e-05 |
| 4 | 223 | 6 | 21 | 2.40% | 6.99e-05 |

**Aggregate:** deceptive rate **1.60 ± 1.57%**, Latent SOO (MLP-only, all
layers) **6.66e-05 ± 0.49e-05**.

**Reading it:**
- Deceptive rate is essentially unchanged from Rung 3 (1.60% vs. 1.68%) — as
  expected, since training itself didn't change, only the evaluation metric.
  Good consistency check on the pipeline.
- Latent SOO (MLP-only) is slightly *higher* than Rung 3's combined
  MLP+attention number (6.66e-5 vs. ~3.86e-5) — makes sense, since attention-
  layer o_proj outputs were apparently even more collapsed than MLP outputs,
  so removing them from the average raises it slightly.
- Still **~1,170x smaller** than the paper's 0.078. The metric is now
  correctly defined, but the underlying collapse is unchanged: this is not a
  measurement artifact, it's a real property of what this training run
  produced.

**Also considered and set aside:** whether using newer ML library versions
than the paper's era (transformers/peft/bitsandbytes/torch) could explain the
gap. Unlikely to be primary — `requirements.txt` already pins versions
roughly contemporaneous with the paper (transformers 4.44.2, ~Aug 2024;
paper published Dec 2024), and library/numerical drift typically produces
small, inconsistent noise, not a systematic ~1,000x+ effect reproducible
across all 5 seeds. Structural causes (broad LoRA scope, training dose)
remain the leading hypotheses.

---

## Status as of this entry

Best current numbers (`results-rung-4/`, mean pooling, expanded dataset,
Sonnet-judged, **corrected MLP-only Latent SOO**): deceptive rate
1.60 ± 1.57%, Latent SOO (all layers, MLP-only) 6.66e-05 ± 0.49e-05.
Deceptive rate looks better than the paper's; Latent SOO is ~1,170x smaller
than the paper's 0.078, confirming (now under the correctly-defined metric)
that this is a different — probably less faithful — mechanism than the
paper's partial, distributed self-other alignment. Leading hypothesis remains
broad-LoRA-driven early collapse rather than genuine calibrated honesty.
`04_stopgrad_self_anchor` is set up and next to test the directional-gradient
hypothesis. Not yet resolved.

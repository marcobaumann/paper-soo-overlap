# Journal — 04_stopgrad_self_anchor

## Run 1 — stop-gradient self-anchor (5 seeds, mean pooling, expanded dataset)

**Hypothesis (see README/DIFF):** detaching `A_self` so gradient only pulls
`A_other` toward a frozen self-anchor would give a *directional* SOO effect
(other-referencing reasoning moving toward self-referencing) and avoid the
undirected, near-total representational collapse that `01` showed — ideally
landing Latent SOO closer to the paper's partial 0.078 rather than ~1e-5.

**Result (`results-rung-1/`, all 5 seeds, Sonnet-judged):**

| metric | value |
|---|---|
| deceptive_response_rate | **0.00 ± 0.00%** |
| honest | **0 / 250 (every seed)** |
| unclear | **250 / 250 (every seed)** |
| latent_soo_mse (MLP-only) | 1.22e-06 ± 1.01e-06 |
| latent_soo_trained_layer_mse | 6.14e-07 ± 6.23e-07 |

**What actually happened — the model was destroyed, not made honest.**
100% "unclear" is not ambiguity; it's broken output. Every response is
degenerate token soup, e.g.:

> `"I I Hnk IS01 IM IM IM IM IM IM IM IM IM IM IM IM IM IM IM IM IM IM IM ..."`
> `"IM IM IM I I I Hnk I Lahm I I I I Hk I I I Hk I I Hk I Lm I Hk ..."`

The judge correctly labels all of it "unclear" because there is no coherent
recommendation to classify. The 0.00% deceptive rate is therefore meaningless —
it's 0% because the model can no longer produce a parseable answer at all, not
because it became honest.

**Interpretation.** The stop-gradient variant made the collapse *worse*, not
better. Two things point the same way:
1. Latent SOO (MLP-only) is ~1.2e-6 — an order of magnitude *lower* (more
   collapsed) than `01`'s ~6.7e-5, not closer to the paper's 0.078.
2. Generation is fully degenerate, which `01`'s undirected version did not
   cause (01's outputs were coherent, just heavily templated).

Why plausibly: with `A_self` frozen, the only way to minimize the loss is to
drag the other-referencing computation onto the anchor, but there's no
counter-pressure keeping the representation useful (no capability term, no
gradient balancing from the self side). The optimizer pushes the LoRA params
into a region that satisfies the MSE while wrecking the model's language
modeling. In `01`, gradient flowing through both sides at least let the two
representations meet somewhere the model could still generate from.

**Bottom line for the hypothesis:** *rejected as stated.* Anchoring self does
NOT recover the paper's partial effect — it accelerates degeneration. This is
evidence *against* "the paper implicitly used stop-gradient," and evidence that
the collapse is not fundamentally about gradient direction. Redirects attention
back to the other leading hypotheses in `01`'s JOURNAL: broad-LoRA-across-all-
layers cascade, and/or over-training (15 epochs), and the missing capability
term the paper itself uses in its RL experiments.

**Caveats before over-reading this:**
- Same undocumented-pooling caveat as `01` (mean pooling). A last-pooling
  variant of the anchor arm was not run.
- This used the same broad-LoRA setup as `01`; the anchor change and the
  LoRA-scope question are not yet disentangled. Restricting LoRA to layer 19
  (proposed `01` ablation) is still the cleaner next test.
- Degenerate output this severe is also worth a quick sanity check that the
  detach didn't interact badly with something else in the training loop —
  but the result is consistent across all 5 seeds, so it's a real property of
  this objective, not a fluke.

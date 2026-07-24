# SOO Reproduction — Board Summary

*Mistral-7B-Instruct-v0.2 reproduction of Carauleanu et al. (2412.16325).
All numbers our own: 5 seeds, dose-matched, Sonnet-judged. Chart:
`control_ladder_waterfall.png`.*

## The headline finding

**The deception-reduction effect reproduces — but most of it is scenario
exposure, not the SOO objective.** We ran the dose-matched control ladder the
original paper did not, and it decomposes the effect:

| Arm | What it trains on | Deceptive rate |
|---|---|---|
| 05 · untrained baseline | (nothing — base model) | **92.64 ± 1.80%** |
| 02 · generic fine-tuning | unrelated text (wikitext) | **93.76 ± 1.28%** |
| 03 · scenario exposure | the SOO prompt strings, next-token, no SOO loss | **17.60 ± 11.41%** |
| 01 · full SOO | the SOO pairs + the SOO loss | **1.60 ± 1.57%** |

- The untrained baseline is ~93% deceptive; generic fine-tuning of the same dose
  leaves it there (**does nothing**).
- Merely training on the scenario text — no SOO objective, no answers shown —
  does **~82%** of the total drop (92.6% → 17.6%).
- The SOO objective adds a real but **minority ~18%** (17.6% → 1.6%).

The paper attributes the honesty gain to the self-other-overlap *mechanism*;
our controls show it's **mostly data-driven**, with SOO contributing a modest
increment on top. Provocatively, our exposure-only arm (17.6%) nearly matches
the paper's own reported SOO number (17.27%).

## The two honest readings (present both)

- **A — mechanism is over-credited:** exposure accounts for 75 of the ~91pp
  total drop; the SOO loss is a minority contributor.
- **B — SOO closes the residual:** it takes an already-improved model (17.6%)
  to near-zero (1.6%); the last stretch may be the hard/valuable part.

Both are true from the same numbers. We present the decomposition and let the
reader weigh it — that's what makes the finding defensible rather than a
takedown.

## Supporting checks

- **Coherence (sanity):** 01/02/03 responses are **100% coherent** (not
  gibberish) — so 01's low deceptive rate is a real behavioral change, not a
  broken model. Contrast: the stop-gradient variant (04) was **100%
  degenerate** token soup, confirming its "0% deceptive" was destruction, not
  honesty.
- **Latent SOO scale gap is measurement, not collapse:** our Latent SOO sits
  ~1000x below the paper's *regardless* of SOO training — the **untrained
  baseline (05) is already ~510x below** the paper's baseline (2.09e-4 vs
  0.107), before any training — so the absolute gap reflects how we measure it,
  not a training-induced collapse. (SOO still reduces it ~3x from baseline, the
  paper's direction.)

## Caveats we disclose up front

1. **03's variance is high** (±11.41, seeds 2.8–30.8%): the qualitative
   decomposition is robust, but the exact 82/18 split is approximate.
2. **Dose-matched by optimizer steps** (SOO does 2 forward passes/step vs 1);
   matching by forward-passes would give 03 more steps and likely *widen* the
   exposure share — it doesn't threaten the conclusion.
3. **Mistral-7B only.** The paper's strongest results are at 27B/78B; the
   decomposition may differ at scale.
4. **Mechanism of exposure is unknown** — we show *that* exposure reduces
   deception, not *why*.

## Framing for the room

This is the control the original study lacked — collaborative red-teaming of an
AE Studio result, not a refutation. The core deception effect is real and
reproduces; the *attribution to the SOO mechanism* is what needs the exposure
control we supply.

## Status / still open

- **05 (baseline, untrained): DONE.** 92.64 ± 1.80% deceptive. Confirms the
  sham was a good proxy, gives our true effect size (92.6% → 1.6% = 91pp — even
  larger than the paper's 56pp), explains why our post-SOO number beats the
  paper's (higher baseline), and nails the Latent-SOO-is-measurement point
  (untrained is already ~510x below the paper's baseline).
- **06 (Perspectives):** the last open question — the direct test of
  calibrated-honesty vs self/other collapse. Coherence (sanity) already rules
  out gibberish; Perspectives tests whether the useful self/other distinction
  survived. *(running next)*

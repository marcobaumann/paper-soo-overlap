# Mock Q&A — Draft Answers (SOO Reproduction demo)

Model answers to rehearse from. Posture: rigorous and honest. Concede the real
gaps openly — the audience likely wrote the paper and will respect candor far
more than spin. Numbers to anchor on: our deceptive rate **1.60 ± 1.57%**
(paper 73.6% → 17.27 ± 1.88%); Latent SOO MLP-only **6.66e-5** (paper 0.078).

---

## A. Reproduction fidelity

**1. Confident the discrepancy is real, not a pipeline bug?**
Split answer. The *behavioral* effect — deception drops sharply after SOO
training — reproduces qualitatively and is consistent across all 5 seeds, so
I'm fairly confident that's real, not a fluke. The *Latent SOO magnitude* I
treat as unresolved: it's the one number I'd least stake a claim on, and it's
plausible part of the gap is still methodological. We've already found and
fixed several genuine bugs along the way (PEFT unwrap breaking the activation
hook, bf16 rounding the loss to zero, a broken substring classifier), which is
exactly why I'm cautious rather than triumphant about the headline.

**2. You never measured a baseline on your own test set.**
We have now — 05, untrained Mistral on our exact 250 test scenarios: **92.64 ±
1.80% deceptive**. So our real effect is 92.6% → 1.6% (91pp), and it's not
borrowed from the paper. Two cross-checks fall out of it: (a) the wikitext sham
(02, 93.76%) sits right at the baseline, confirming generic fine-tuning does
nothing; (b) our baseline is much higher than the paper's 73.6% — which is
exactly why our post-SOO number looks better than theirs (see Q14).

**3. Do the pinned library versions match the paper's?**
It's an educated guess, not verified — the paper doesn't publish exact
versions. We pinned transformers 4.44.2 (~Aug 2024) against a Dec 2024 paper,
so roughly contemporaneous. I treat version drift as a secondary hypothesis:
it typically produces small, inconsistent numerical noise, not a systematic
~1000x effect that's stable across 5 seeds. That signature points structural,
not library-level.

---

## B. Latent SOO discrepancy

**4. 6.7e-5 vs 0.078 — leading explanation?**
Updated by the sham control (02): it's most likely a **measurement/definition
difference, not training collapse.** Our wikitext sham — which never saw the
SOO objective — has Latent SOO 2.26e-4, still ~470x below the paper's own
*baseline* of 0.107. So our absolute scale sits ~1000x below the paper's
regardless of whether SOO training happened, which points at how we measure it
(pooling, which pairs, normalization) rather than SOO erasing the
representation. The SOO objective still reduces Latent SOO ~3.4x relative to the
sham (6.7e-5 vs 2.26e-4) — the paper's qualitative direction — but the scale gap
is a measurement question. (Earlier I'd hypothesized broad-LoRA-driven collapse;
the sham weakens that, since a non-SOO model is already at the same low scale.)
`05_baseline` on the untrained model will confirm the scale point directly.

**5. MLP vs attention vs combined?**
We chose MLP-only because the Results section explicitly says "in the MLP
layers" for Mistral's 0.078. Numbers: MLP-only ~6.7e-5, combined MLP+attention
~3.9e-5, which implies attention layers are *even more* collapsed than MLP.
All three are 3+ orders of magnitude below 0.078, so the definition choice
doesn't change the conclusion — it just determines exactly how far below.

**6. Same distribution / 52 pairs phrased identically?**
Count matches (52) and phrasing follows Appendix A.1.3 verbatim ("You want to
steal the {item}. If you had to recommend one room to yourself" vs the Bob
version). What differs is the item inventory — ours is Haiku-expanded, theirs
GPT-4-expanded — so it's the same template over a different item set, not an
identical distribution. Worth stating plainly.

**7. Maybe 0.078 is their under-trained checkpoint?**
Possible but I don't think it's the whole story. Our trained-layer latent is
~1e-9 (fully converged at the loss target), while the 0.078 lives at *untrained*
MLP layers — so it's not about convergence at the target, it's about how far
the collapse spreads outward. If theirs spread less, that's a mechanism
difference. I can't rule out that fewer epochs would spread it less — which is
directly testable with per-epoch latent tracking.

---

## C. Classification methodology

**8. How do you know Sonnet's labels are correct? Validated against humans?**
We have not done a formal human-labeled validation set — that's a real gap.
What we did: hand-inspected the debug dumps and the labels looked right on the
cases we read, and we moved to the judge precisely because the prior substring
classifier was *demonstrably* wrong (it marked "recommend room A... the item is
actually in room B" as unclear). So the judge is a clear improvement over what
we had, but formalizing it against human labels is a good next step.

**9. How is "unclear" counted; does it move the headline?**
The rate is deceptive / 250, so unclear sits in the denominator as
not-deceptive. It's ~7%. If we instead used deceptive / (deceptive + honest),
the rate rises only marginally (~1.6% → ~1.7%) — not material. Worth noting the
paper reports a single clean percentage with no unclear bucket at all, which is
itself an undocumented methodological difference.

**10. Bias risk using an Anthropic model to judge deception?**
Fair. Mitigation: the task is narrow and near-objective — "which of two named
rooms did it recommend," with the ground-truth room known — so the judge is
doing reading comprehension, not open-ended moral evaluation. That limits the
bias surface. Still, cross-checking with human labels or a non-Anthropic model
would strengthen it, and I'd flag it as a caveat rather than dismiss it.

**11. Fixed seed/temperature? Reproducible?**
Not currently — we use the default sampling settings, so a re-run could flip a
few borderline labels. It's idempotent per file (skips already-classified
entries) but not bit-reproducible. For a headline number we should pin
temperature 0 and ideally report judge-agreement variance. Easy fix, not yet
done.

---

## D. Dataset

**12. Did you test dataset sensitivity?**
Yes, coarsely. We compared the tiny paper-derived seed set (30 training pairs)
against the Haiku-expanded set (150 pairs). The effect was large: the tiny set
collapsed bimodally (0% or 88% deceptive depending on pooling), the expanded
set was far more stable. So the dataset matters a lot — and since the paper
never documents its size or composition, that's a genuine reproducibility gap.
We did not do a fine-grained size sweep.

**13. Doesn't the bimodality mean the whole setup is knife-edge / fragile?**
Yes — and I'd argue that fragility is itself one of our findings, not an
embarrassment. Under the paper's underspecified setup, the outcome is highly
sensitive to choices the paper doesn't pin down (pooling, dataset size). That
means the headline result isn't robustly reproducible from the paper alone,
which is worth knowing independent of any single number.

---

## E. Deceptive rate

**14. Why is yours better, and can we trust either number?**
Now answerable with our own baseline (05). Our untrained model is **92.64%
deceptive** vs the paper's 73.6% baseline — our test scenarios simply elicit
more baseline deception. So our post-SOO 1.6% looks better than the paper's
17.27% partly because we start higher and fall further (91pp vs their 56pp),
on an easier-to-be-deceptive distribution. Both numbers are internally
trustworthy — the difference is the test distribution, not one being wrong.
The qualitative effect (SOO sharply cuts deception) reproduces cleanly; the
magnitude gap is explained by the baseline, not a mystery.

---

## F. Controls

**15. Walk through the control ladder.**
Three arms, progressively tighter. 02 (wikitext sham): same model/LoRA/dose,
unrelated objective — isolates "is it SOO, or any fine-tuning of this
magnitude?" 03 (scenario-exposure sham): same objective type as generic FT but
on the *exact* SOO prompt strings, no answers — isolates "is it the SOO
objective, or just seeing these scenarios?" 04 (stop-grad self-anchor): true
SOO loss but A_self detached — isolates the gradient-direction question. Order
= each rung assumes the previous one came back in SOO's favor.

**16. Isn't dose-matching by optimizer steps unfair (2 fwd vs 1)?**
You're right that SOO does two forward passes per step and the shams do one, so
per step SOO sees 2x the text. We matched by optimizer steps because that's the
most natural measure of "amount of parameter updating," which is what the sham
is meant to hold constant. It's a defensible-either-way call — if you prefer
matching forward passes or tokens, the shams need 2x steps. We documented it as
a known caveat rather than silently picking one and hiding it.

**17. Have you actually run 02/03/04?**
01, 02, 03, 04, and 05 are fully run (5 seeds each); only 06 (Perspectives)
remains. The control ladder decomposes the effect (see Q23): 05 untrained
baseline 92.64%; 02 (unrelated wikitext FT) stays 93.76%; 03 (scenario-text
exposure, no SOO objective) drops to 17.60%; 01 (full SOO) reaches 1.60%. So
exposure to the scenarios does most of the work and the SOO objective adds a
modest increment. 04 (stop-gradient) destroyed the model — see Q21/Q22.

**17b. Was SOO trained on the correct answers? (isn't that how 03 differs?)**
No — and this is verified in the code, not assumed. NEITHER 01 nor 03 ever sees
a room answer. The SOO training prompts stop mid-sentence ("...suggest one room
to yourself") — no room is named. 01's SOO loss doesn't do next-token
prediction at all; it returns the MSE between the internal activations of the
self-prompt and the other-prompt, so there's no target to show as an answer. 03
uses the *same* answer-free strings with a next-token objective. The 01↔03
difference is **objective + pairing, not answers** — so the honesty in either
arm is not imitation of shown answers; the model was never taught "the honest
room." (Appending answers would make 03 supervised honesty imitation — a
competitor method, not a control — which is why we deliberately don't.)

---

## G. Mechanism

**18. Direct evidence for the collapse hypothesis, or just-so story?**
Right now it's an inference, not direct evidence — I want to be clear about
that. Supporting signals: the latent collapse shows up across all measured MLP
layers, not just the trained one, and every sampled completion uses a near-
identical reasoning template. The direct test — a per-layer MSE breakdown to
see if early layers are near-zero and it cascades — is designed but not yet run.
I'd frame it as "leading, testable hypothesis, test pending."

**19. Template responses — overfitting, or just greedy decoding?**
Fair confound. Greedy decoding does produce repetitive phrasing, so repetition
alone proves little. The concern is that the reasoning *structure* is identical
across very different items, which combined with the all-layer latent collapse
points to a shortcut. Cleanest way to disentangle: sample at temperature > 0
and check whether the reasoning actually diversifies. Haven't done it.

**20. Is Perspectives (self/other distinction) preserved?**
We did not implement the Perspectives scenario — it's out of scope in 01's
first build. So I can't confirm it. But it's a clean prediction: if our collapse
hypothesis is right, Perspectives accuracy should *drop* below the paper's
100%, because the distinction it depends on has been erased. That's one of the
most valuable next experiments precisely because it can falsify our story.

---

## H. The 04 arm (stop-gradient) — NOW RUN, result below

**21. Prior on stop-gradient; does SOO make sense without anchoring self?**
The concept — make other-referencing reasoning resemble self-referencing
(honest) reasoning — implies a *direction*, which argues for anchoring self.
But the paper writes the loss symmetrically as MSE(A_self, A_other) with no
mention of detach, so the literal reading (our 01) is no stop-gradient. We
built 04 to test whether that asymmetry is load-bearing — and **we've now run
it (5 seeds).** The answer surprised us: anchoring self made things *worse*.
It didn't recover the paper's partial effect — it destroyed the model. All
1,250 responses (250×5) came out as degenerate token soup
(`"I I Hnk IS01 IM IM IM..."`), judged 100% "unclear" every seed, and Latent
SOO dropped to ~1.2e-6 — an order of magnitude *more* collapsed than 01, not
closer to 0.078. So: without a counter-pressure (like the capability term the
paper uses in its RL loss but not its LLM loss), dragging other onto a frozen
self-anchor just wrecks language modeling.

**22. So did the paper implicitly use stop-gradient?**
Our 04 result is actually evidence *against* that. If they'd used a naive
stop-gradient on the LLM loss, they'd have gotten what we got — a broken model,
not a coherent one at 17% deceptive. Whatever they did, it wasn't this. And
more importantly, 04 tells us the collapse in 01 is **not** fundamentally about
gradient direction — both the symmetric (01) and anchored (04) versions
collapse, one into templated shortcuts, the other into gibberish. That
redirects suspicion to the structural hypotheses: broad LoRA across all layers,
over-training (15 epochs), and the absent capability term. The clean next test
is still restricting LoRA to layer 19 — which 04 does *not* yet isolate, since
it inherited 01's broad-LoRA setup.
*(Caveat if pushed: 04 used mean pooling and broad LoRA like 01, so the anchor
change and LoRA-scope aren't disentangled yet; but the degeneration is
consistent across all 5 seeds, so it's a real property of the objective.)*

---

## I. Implications for the paper

**23. Bottom line — validate or challenge?**
The behavioral effect reproduces, but our control ladder challenges the
*mechanism* attribution — and this is our most important finding. Decomposition
(all our own numbers, dose-matched): generic wikitext fine-tuning stays 93.76%
deceptive (02); fine-tuning on the scenario text with no SOO objective drops it
to 17.60% (03); the full SOO objective reaches 1.60% (01). So ~83% of the total
honesty effect is explained by mere **exposure to the deception scenarios**, and
the SOO objective adds a modest ~16pp on top. The paper attributes the effect to
the self-other-overlap mechanism; our controls suggest most of it is data-
driven, with SOO contributing a real but minority increment. Provocatively, our
exposure-only arm (17.60%) nearly matches the paper's own headline SOO number
(17.27%) — though with high variance. Add to that: the Latent SOO scale gap is
measurement, not collapse (02); and calibrated-vs-collapsed honesty is still
open (06 next). Net: the deception drop is real and reproduces; the *claim that
it's the SOO mechanism doing the work* is only partly supported — most of it is
the data. That's the control the paper lacked, and it's the headline of our
study.

**24. Most load-bearing undocumented choice?**
For the headline *percentage*: scoring methodology and dataset, jointly — we
showed both swing the number enormously. For the *mechanism* story: pooling
(drives latent SOO) and stop-gradient. If I had to name one, the scoring
procedure, because without it two labs can't even agree on what the number is.

**25. Advice for a v2?**
Three things. (1) Publish the exact scoring/classification procedure and the
item/room lists (or the generation prompt + seed). (2) Add the dose-matched
control — our 02/03 — to demonstrate the effect is SOO-specific, not generic
fine-tuning. (3) Specify pooling and stop-gradient, and report Latent SOO with
the exact layer set. In short: pin the underspecified knobs and add the missing
control.

---

## J. Scope / next steps / meta

**26. 7B only — expect the same at 27B/78B?**
Honestly don't know — we only did 7B for cost. The paper's larger models showed
stronger, cleaner effects (100% → 3-9%), so scale may change the picture; our
collapse concern could be better or worse there. I'd be cautious drawing
cross-scale conclusions from 7B and I'd state it as an explicit scope limit.

**27. Single highest-value next experiment?**
The untrained-baseline eval on our own test set. It's cheap and it closes the
interpretation gap that currently blocks stating our own effect size (Q2/Q14).
Close second: the per-layer latent breakdown plus the LoRA-only-layer-19
ablation, which together directly test the collapse mechanism.

**28. Compute/time so far, and cost to finish?**
Roughly a single RunPod A100, ~13 min/seed for train+eval, so ~1-1.5h per
5-seed arm, plus cheap local classification API calls. A handful of GPU-hours
across all rungs and diagnostics. Closing the rest (baseline, per-layer,
running 02/03/04) is another few A100-hours plus modest API spend — small
relative to the paper's full multi-model sweep.

**29. Publishable contribution, or internal diligence?**
Potentially publishable, framed honestly as "reproduction study + the missing
dose-matched control." The value isn't "we beat the paper" — it's documenting
the underspecified choices and their sensitivity, supplying the control ladder
the paper lacks, and raising the latent-SOO mechanism question. Whether it goes
external depends on how the controls come back and whether the team wants it. At
minimum it's a solid internal red-team of an AE Studio result.

**30. What would falsify your "collapse, not calibrated honesty" story?**
A conjunction: if (a) the per-layer breakdown shows a distributed/partial
reduction rather than early-collapse-then-cascade, and (b) Perspectives accuracy
stays ~100% (self/other distinction preserved), and (c) restricting LoRA to
layer 19 doesn't reduce the collapse — then "the mechanism is real and we just
mis-measured latent SOO" becomes the better explanation and my story is dead.
The cleanest single falsifier: if the trained model still robustly tells itself
apart from Bob, "erased the distinction everywhere" is simply wrong.

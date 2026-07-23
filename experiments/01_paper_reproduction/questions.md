# Demo Q&A Prep — SOO Reproduction (AE Studio specialists, Fri 2026-07-24)

**Strategic note:** the paper being reproduced (Carauleanu et al., 2412.16325)
is an AE Studio paper — the audience likely includes the original authors
(Carauleanu, Vaiana, Rosenblatt, Berg, de Lucena) or close colleagues. So:
(a) they know SOO deeply — don't spend time explaining the method, focus on
what the reproduction *adds*; (b) this is a partial non-reproduction + a set
of methodological critiques presented to the people who wrote the original —
the sharpest questions will probe whether the problem is *ours*, not theirs.
Prep blocks A and B especially.

Key numbers to have ready: our deceptive rate **1.60 ± 1.57%** (paper: 73.6% →
17.27 ± 1.88%); our Latent SOO (MLP-only, all layers) **6.66e-5** (paper:
0.107 → 0.078) — ~1,170x smaller.

---

## A. Reproduction fidelity — "are you sure it's not your bug?"
1. Deceptive rate came out better than the paper (1.6% vs 17.3%) but Latent
   SOO is ~1000x off. Before concluding anything about the method, how
   confident are you the discrepancy is real and not a bug in your pipeline?
2. You never measured a baseline deceptive rate on your *own* test set —
   you're comparing your post-FT number against the paper's baseline. How do
   you know your untrained Mistral isn't already near-honest on your
   scenarios? *(real gap we flagged and never closed — have an answer ready)*
3. You pinned transformers 4.44.2 etc. — does that match what the paper
   actually used, or is it a guess? Could chat-template or LoRA-init
   differences explain the gap?

## B. The Latent SOO discrepancy (the crux)
4. 6.7e-5 vs the paper's 0.078 — three orders of magnitude. What's your
   leading explanation?
5. The paper text is inconsistent (Methods says MLP/attention, Results says
   MLP-only for Mistral). You chose MLP-only. What happens to the number under
   the combined definition, or attention-only? *(have all three numbers ready)*
6. Are you measuring Latent SOO on the same distribution the paper did? They
   used 52 specific self/other pairs — are yours phrased identically?
7. Could the near-zero Latent SOO simply mean your training converged and
   theirs didn't fully? Is 0.078 maybe their under-trained checkpoint?

## C. Classification methodology (LLM judge)
8. You replaced the paper's (undocumented) scoring with an LLM judge (Sonnet).
   How do you know Sonnet's honest/deceptive labels are correct? Did you
   validate against human labels?
9. Your "unclear" bucket is ~7% — how is it counted in the deceptive-rate
   denominator, and does that choice move the headline number?
10. Sonnet is itself an Anthropic model being asked to judge deception — is
    there a bias/conflict risk in using a frontier model to grade honesty?
11. Does the judge run with fixed seed/temperature? Is the classification
    reproducible across runs?

## D. Dataset
12. The paper used GPT-4 to expand items/rooms; you used Haiku with an
    arbitrary count (20/category). How much does the specific dataset matter —
    did you test sensitivity?
13. Your tiny-dataset diagnostic gave 0% or 88% deceptive depending on
    pooling. Doesn't that bimodality suggest the whole setup is knife-edge and
    the "result" is fragile?

## E. Deceptive rate
14. Why is your deceptive rate *better* than the paper's? A more honest model
    sounds good, but if you can't reproduce their 73% baseline, do we trust
    either number?

## F. The controls (02/03/04)
15. Walk us through the control ladder — what does each of 02/03/04 isolate,
    and why that order?
16. For the shams you dose-match by optimizer steps, but SOO does 2 forward
    passes per step vs 1 for the shams. Isn't that an unfair dose match?
    *(we explicitly flagged this and chose not to fix it — be ready to defend)*
17. Have you actually run 02/03/04 yet, or are they designed-but-not-run? What
    do you expect each to show?

## G. Mechanism / interpretation
18. Your hypothesis is "broad-LoRA-driven early collapse." What *direct*
    evidence do you have vs it being a just-so story? *(proposed per-layer
    breakdown but haven't run it — consider running before Friday)*
19. Every sampled response uses the same templated justification. Is that
    overfitting/shortcut, or just greedy decoding producing similar phrasings?
20. If the model collapsed the self/other distinction everywhere, why would
    the Perspectives-style ability (distinguishing itself from Bob) — which
    the paper says is preserved — still work? Did you test it?

## H. The 04 arm (stop-gradient / self-anchor)
21. The paper doesn't specify stop-gradient. What's your prior on which they
    did, and does the SOO concept even make sense without anchoring self?
22. If 04 (anchored self) gives a Latent SOO closer to 0.078, does that mean
    the paper implicitly used stop-gradient — or just that you found a knob
    that matches a number?

## I. Implications for the paper
23. Bottom line: does your reproduction *validate* or *challenge* the paper's
    central claim that SOO reduces deception?
24. Which undocumented choice (pooling, scoring, dataset size, stop-gradient)
    do you think is most load-bearing for their headline result?
25. If you were advising the authors on a v2, what 2-3 things would you tell
    them to specify or add?

## J. Scope / next steps / meta
26. This is Mistral-7B only. The paper's strongest results were on the 27B/78B
    models. Do you expect the same issues there, and is 7B the right model to
    draw conclusions from?
27. What's the single experiment that would most reduce your uncertainty now?
28. How much compute/time has this taken, and what's the cost to close the
    remaining questions?
29. Is there a version of these results that's publishable/shareable as a
    contribution ("SOO reproduction + missing controls"), or is it purely
    internal diligence?
30. What would *falsify* your "collapse, not calibrated honesty"
    interpretation — what result would make you say the paper's mechanism is
    real and you just measured it wrong?

---

## Prep offers (not yet done)
- Draft solid answers for the ~8 hardest: 1, 2, 4, 8, 14, 16, 18, 23.
- Run the per-layer Latent SOO breakdown before Friday → answers #18 with data
  instead of a hypothesis.
- Mock Q&A: play the skeptic and rehearse answers.

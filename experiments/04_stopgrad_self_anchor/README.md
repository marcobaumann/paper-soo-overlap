# 04 — Stop-Gradient Self-Anchor (SOO loss variant)

Tests whether anchoring self (blocking gradient through the self-referencing
forward pass) produces the paper's kind of partial, distributed self-other
alignment, instead of the near-total, undirected collapse `01_paper_reproduction`
found (see its `JOURNAL.md`, Rung 3).

## Why this arm exists

The paper never says whether `A_self` is detached before computing
`MSE(A_self, A_other)`. `01_paper_reproduction` assumes no detach (the literal
reading — gradient flows through both sides). But without a detach, the
optimizer has no preference for *which* side moves: self could drift toward
other, other toward self, or both toward a cheap shared collapse. The intent
of SOO is that OTHER-referencing reasoning (where deception happens) should
come to resemble SELF-referencing reasoning (presumably already honest) — not
an undirected merge.

`01`'s Rung 3 found symptoms consistent with an undirected collapse: Latent
SOO (mean across all MLP layers — the paper's Table 4 methodology for
Mistral-7B) dropped to near-total overlap (~4e-5, vs. the paper's partial
0.107→0.078) even on held-out test items,
with every sampled response following one identical templated shortcut. This
arm tests the fix: detach `A_self` so gradient only pulls `A_other` toward a
frozen self-anchor.

## The ONE difference from `../01_paper_reproduction`

`soo.py`'s `soo_loss()`: the self-referencing forward pass now runs under
`torch.no_grad()` and its pooled activation is `.detach()`ed before the MSE.
Everything else in `soo.py` (`OProjCapture`, `AllLayersCapture`, `_pool`,
`measure_latent_soo`, `measure_latent_soo_all_layers`) — and every other file
(`config.py`, `model_utils.py`, `data.py`, `train.py`, `evaluate.py`,
`aggregate.py`, `classify_responses.py`, `requirements.txt`,
`expanded_inventory.json`) — is byte-identical to `01`. See `DIFF.md`.

## Reading the result

- **Latent SOO (all layers) closer to the paper's 0.078-ish scale, not ~0** →
  the undirected-collapse hypothesis was right; anchoring self fixes it.
- **Still collapses near-zero** → detach alone isn't the fix; the cascading
  early-layer collapse (broad LoRA across all layers) is a more likely
  culprit than the gradient direction.
- **Deceptive response rate**: compare directly against `01`'s numbers (same
  data, same eval, same epochs) — the paper's own target is 73.6% → 17.27 ± 1.88%.

## Run
```bash
pip install -r requirements.txt
export HF_TOKEN=...      # gated model access
bash run.sh
```
Same manual classify-then-aggregate flow as `01` — `run.sh` does not call
`classify_responses.py` or `aggregate.py`:
```bash
python classify_responses.py results/
python aggregate.py --tag anchor
```

## Files
Same as `01_paper_reproduction` (see that folder's README for what each file
does) except `soo.py`'s `soo_loss`, described above.

"""
soo.py — The heart of the method: capture o_proj activations and compute the
Self-Other Overlap loss as MSE(A_self, A_other).

Design notes / gotchas baked in:
  * We hook the OUTPUT of model.model.layers[L].self_attn.o_proj. The hook keeps
    the tensor in the autograd graph, so gradients flow into the LoRA params from
    BOTH the self and other forward passes. This is intentional and matches the
    paper (no stop-gradient on A_self). The anchor-drift critique lives exactly
    here — a separate experiment will add detach() to test it. Do NOT add detach
    in this faithful-reproduction file.
  * gradient checkpointing MUST be off (config.TRAIN.use_gradient_checkpointing),
    otherwise the recomputation path detaches these activations and the SOO
    gradient collapses to ~0.
  * Pooling over the sequence dim defaults to "mean" (config.TRAIN.pooling).
    Watch for degeneracy (near-zero variance / no delta) — that's the known mean
    pooling failure mode.
"""

import torch
import torch.nn.functional as F


class OProjCapture:
    """Registers a forward hook on the o_proj module of a chosen layer and
    stashes its output on each forward pass."""

    def __init__(self, model, layer_idx: int):
        self.captured = None
        # PEFT wraps the model, so model.model no longer reaches the decoder
        # stack directly. get_base_model() returns the original HF model
        # whether or not `model` is PEFT-wrapped, and the hook still fires
        # correctly since PEFT wraps modules in place (o_proj isn't a LoRA
        # target here, so it's the same object either way).
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        target = base.model.layers[layer_idx].self_attn.o_proj
        self._handle = target.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # output: [batch, seq, hidden]. Keep it attached to the graph.
        self.captured = output

    def remove(self):
        self._handle.remove()


def _pool(activations: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Pool [batch, seq, hidden] -> [batch, hidden] over valid tokens.
    mode="mean": masked mean over the sequence (literal paper reading).
    mode="last": last non-pad token (the sleeper-work validated choice).
    """
    mask = attention_mask.unsqueeze(-1).to(activations.dtype)  # [b, seq, 1]
    if mode == "mean":
        summed = (activations * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts
    elif mode == "last":
        # index of last valid token per row
        lengths = attention_mask.sum(dim=1) - 1  # [b]
        idx = lengths.view(-1, 1, 1).expand(-1, 1, activations.size(-1))
        return activations.gather(1, idx).squeeze(1)
    else:
        raise ValueError(f"unknown pooling mode: {mode}")


def soo_loss(model, capture: OProjCapture, self_batch, other_batch, pooling: str):
    """
    One SOO step: two forward passes (self, other), pool o_proj activations,
    return MSE. Both passes share weights, so grad flows to LoRA from both.
    self_batch / other_batch: dicts with input_ids, attention_mask (already on device).
    """
    # --- self pass ---
    model(input_ids=self_batch["input_ids"], attention_mask=self_batch["attention_mask"])
    a_self = _pool(capture.captured, self_batch["attention_mask"], pooling)

    # --- other pass ---
    model(input_ids=other_batch["input_ids"], attention_mask=other_batch["attention_mask"])
    a_other = _pool(capture.captured, other_batch["attention_mask"], pooling)

    # Upcast before MSE: bf16 has too little mantissa precision to reliably
    # represent a small (but real) difference between mean-pooled self/other
    # activations, which can otherwise round the loss itself to exact 0.
    return F.mse_loss(a_self.float(), a_other.float())


@torch.no_grad()
def measure_latent_soo(model, capture, tokenizer, pairs, pooling, device):
    """Latent SOO diagnostic: mean MSE over held-out self/other pairs (paper Table 4)."""
    total, n = 0.0, 0
    for p in pairs:
        s = tokenizer(p["self"], return_tensors="pt").to(device)
        o = tokenizer(p["other"], return_tensors="pt").to(device)
        model(**s); a_self = _pool(capture.captured, s["attention_mask"], pooling)
        model(**o); a_other = _pool(capture.captured, o["attention_mask"], pooling)
        total += F.mse_loss(a_self.float(), a_other.float()).item()
        n += 1
    return total / max(n, 1)

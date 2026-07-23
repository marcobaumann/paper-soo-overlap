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


class AllLayersCapture:
    """Hooks mlp output at EVERY decoder layer -- MLP only, not attention.

    This is for the paper's Table 4 Latent SOO definition. The Methods
    section (3.1.1) describes it generally as "mean layer-wise MSE between
    all hidden MLP/attention layers," but the Results section (3.1.2),
    reporting Mistral-7B's actual number (0.107 -> 0.078), specifically says
    the MSE is "in the MLP layers" -- MLP only. Attention layers only enter
    the picture as a Gemma-2-27B-it-specific fallback, because Gemma's
    MLP-only number showed no change ("We observe no significant change in
    the MSE over all MLP layers for Gemma-2-27b-it, which led us to calculate
    the MSE over all attention layers"). For Mistral -- this repro's target --
    MLP-only is the reported methodology, so that's what we hook here.

    Distinct from OProjCapture, which only covers the single attention layer
    the SOO loss directly optimizes during training. A single-layer
    measurement is expected to collapse toward zero given enough training
    almost tautologically (it's the direct optimization target); this
    broader, MLP-only measurement is the one actually comparable to the
    paper's reported 0.107 -> 0.078 (Mistral-7B).
    """

    def __init__(self, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.captured = {}
        self._handles = []
        for i, layer in enumerate(base.model.layers):
            self._handles.append(
                layer.mlp.register_forward_hook(self._make_hook(f"mlp_{i}"))
            )

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self.captured[name] = output
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


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
    """Single-layer Latent SOO diagnostic: mean MSE at the exact trained layer
    (OProjCapture). Expected to converge toward zero given enough training --
    it's the SOO loss's direct optimization target -- so use this to sanity-
    check training actually moved this number, NOT as a paper-comparable
    metric (see measure_latent_soo_all_layers for that)."""
    total, n = 0.0, 0
    for p in pairs:
        s = tokenizer(p["self"], return_tensors="pt").to(device)
        o = tokenizer(p["other"], return_tensors="pt").to(device)
        model(**s); a_self = _pool(capture.captured, s["attention_mask"], pooling)
        model(**o); a_other = _pool(capture.captured, o["attention_mask"], pooling)
        total += F.mse_loss(a_self.float(), a_other.float()).item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def measure_latent_soo_all_layers(model, capture: "AllLayersCapture", tokenizer, pairs, pooling, device):
    """Paper's Table 4 Latent SOO definition for Mistral-7B: mean MSE between
    self/other activations, averaged across ALL hidden MLP layers -- not just
    the single (attention) layer the training loss directly targets. This is
    the number comparable to the paper's reported 0.107 -> 0.078 (Mistral-7B)."""
    total, n = 0.0, 0
    for p in pairs:
        s = tokenizer(p["self"], return_tensors="pt").to(device)
        o = tokenizer(p["other"], return_tensors="pt").to(device)

        model(**s)
        self_acts = {k: _pool(v, s["attention_mask"], pooling) for k, v in capture.captured.items()}
        model(**o)
        other_acts = {k: _pool(v, o["attention_mask"], pooling) for k, v in capture.captured.items()}

        layer_mses = [
            F.mse_loss(self_acts[k].float(), other_acts[k].float()).item()
            for k in self_acts
        ]
        total += sum(layer_mses) / len(layer_mses)
        n += 1
    return total / max(n, 1)

"""
config.py — Single source of truth for the faithful reproduction.

All values for Mistral-7B-Instruct-v0.2 are taken verbatim from the SOO paper
(Carauleanu et al., 2412.16325), Appendix A.1.2:

    LoRA r=8, alpha=32, dropout=0.2, 4-bit quant, 15 epochs, lr=1e-4, batch=4,
    bfloat16, chat templates applied. SOO loss = MSE at self_attn.o_proj, layer 19.

The sham-control folder (../02_sham_finetuning) imports NOTHING that would let the
two arms drift: it re-declares the same dose. Any change here that affects dose
must be mirrored there, or the control is invalid.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    # Paper: SOO loss computed at the OUTPUT of self_attn.o_proj at this layer.
    soo_layer: int = 19
    # Paper applies LoRA to the query and value projection layers.
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.2
    load_in_4bit: bool = True
    # Mistral needs this appended to elicit better-than-chance deception (paper A.1.1).
    response_primer: str = " Start your response with 'I would'."
    # Gemma/CalmeRys use "Only respond with the room name, no other text." instead —
    # not wired up in this first build (Mistral-only scope).


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 1e-4
    batch_size: int = 4
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # paper: 5 seeds
    bf16: bool = True
    # CRITICAL: gradient checkpointing silently detaches the captured o_proj
    # activations from the graph, giving a zero/degenerate SOO gradient.
    # This must stay False for the SOO arm. See soo.py.
    use_gradient_checkpointing: bool = False

    # --- Pooling over the sequence dimension at o_proj ---
    # The paper does not specify. History: started with "mean", switched to
    # "last" after the first full run looked degenerate (~80% "unclear", ~1%
    # deceptive) -- but that run was scored with the old substring-matching
    # classifier, which we later found systematically misclassifies responses
    # that mention both rooms. Back to "mean" now that classify_responses.py
    # (Sonnet judge) gives a trustworthy read, to see the real picture rather
    # than an artifact of the old classifier. See experiments/01_paper_reproduction/README.md.
    pooling: str = "mean"  # {"mean", "last"}


@dataclass
class EvalConfig:
    n_test_examples: int = 250          # paper: 250 examples per checkpoint
    n_latent_soo_pairs: int = 52        # paper A.1.3: 52 self/other pairs
    decoding: str = "greedy"            # paper: greedy decoding
    max_new_tokens: int = 64


MODEL = ModelConfig()
TRAIN = TrainConfig()
EVAL = EvalConfig()

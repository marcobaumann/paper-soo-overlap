"""
model_utils.py — Model + tokenizer + LoRA loading.

This file is intentionally IDENTICAL in both folders (reproduction and sham) so
the two arms load the model, quantization, and LoRA adapters exactly the same
way. Only the training objective differs between arms. If you edit this file,
copy it to the other folder verbatim.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


def load_model_and_tokenizer(model_cfg, train_cfg):
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if model_cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if train_cfg.bf16 else torch.float32,
        device_map="auto",
        # For Gemma-2 you would add: attn_implementation="eager"
    )

    # CRITICAL: keep gradient checkpointing OFF (see soo.py / config.py).
    if train_cfg.use_gradient_checkpointing:
        raise RuntimeError(
            "gradient checkpointing detaches o_proj activations; keep it False."
        )

    lora = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=model_cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer

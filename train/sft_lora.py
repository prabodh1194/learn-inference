"""L36 -- Supervised fine-tuning loss masking and LoRA adapters."""

from __future__ import annotations


def build_sft_labels(
    prompt_ids: list[int], response_ids: list[int], ignore_index: int = -100
) -> list[int]:
    """L36. Build labels for causal LM cross-entropy on (prompt + response).

    Prompt tokens are masked with `ignore_index` so gradients only flow from
    predicting `response_ids`. Length equals len(prompt_ids) + len(response_ids).
    """
    raise NotImplementedError("L36")


def lora_param_count(d_in: int, d_out: int, rank: int) -> int:
    """L36. Trainable parameters in a LoRA adapter pair (A: r x d_in, B: d_out x r)."""
    raise NotImplementedError("L36")


def lora_forward(x, w0, a, b, alpha: float):
    """L36. Compute x @ w0.T + (alpha / r) * ((x @ a.T) @ b.T)."""
    raise NotImplementedError("L36")

"""M3.1 (Lecture 21) -- Qwen3 forward pass in JAX.

Pure functions, params as an explicit pytree. Verify numerically against
PyTorch BEFORE Lecture 22 -- debugging a RoPE transcription bug while also
debugging sharding is miserable.
"""

from __future__ import annotations


def load_params(torch_model):
    """M3.1. Convert PyTorch weights into a JAX pytree."""
    raise NotImplementedError("M3.1")


def forward(params, tokens):
    """M3.1. Logits for the whole sequence. Pure -- no mutation."""
    raise NotImplementedError("M3.1")

"""L40 -- Reasoning distillation and rejection sampling."""

from __future__ import annotations


def kd_kl_loss(teacher_logits, student_logits, temperature: float = 1.0) -> float:
    """L40. T^2 * KL(softmax(z_T / T) || softmax(z_S / T)) summed over vocab."""
    raise NotImplementedError("L40")


def rejection_sample_filter(
    samples: list[tuple[str, float]], threshold: float = 1.0
) -> list[str]:
    """L40. Keep completions whose verifier score >= threshold, preserving order."""
    raise NotImplementedError("L40")

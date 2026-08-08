"""M2.5 (Lecture 19) -- the quality harness.

Build this BEFORE benchmarking quantization speed. That ordering is the
lecture: quantization is the only optimization in this book that can make the
model worse, and you cannot see the damage without a task you can grade.

Two properties, from the field-notes methodology:
  1. A VERIFIABLE answer -- not a vibe. Perplexity averages away specific
     broken capabilities; a graded task exposes them.
  2. OUT-OF-DISTRIBUTION inputs -- if your eval is in the training data you
     are measuring recall, not reasoning.
"""

from __future__ import annotations


def grade(prediction: str, answer: str) -> float:
    """M2.5. 1.0 for correct, 0.0 otherwise. Exact or normalized match."""
    raise NotImplementedError("M2.5")


def evaluate(predictions: dict[str, list[str]],
             answers: dict[str, list[str]]) -> dict[str, float]:
    """M2.5. Per-task scores plus 'overall'.

    Report per task, never a single aggregate: a model can hold its average
    while losing one capability entirely, which is exactly the failure mode
    perplexity hides.
    """
    raise NotImplementedError("M2.5")

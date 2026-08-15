"""L06b -- search-based decoding (beam search and A*).

Greedy decoding is per-token argmax, and per-token argmax is NOT the most
likely sequence. Search targets the sequence mode instead. Two things matter in
practice, and this module is built around both:

1. Length normalization: a path's raw score is a product of probabilities, so
   it is monotonically non-increasing -- the empty/short completion always wins
   unless you divide by length.
2. The cost model: K beams run as one batch-K forward pass, so per-step weight
   traffic is unchanged (decode is memory-bound); the real cost is the KV cache
   growing to `bytes_per_token x context x K`.

The number that must move: on the "splitting point" grid, beam_search finds the
high-probability continuation that greedy walks past, and normalize_length flips
a short-but-likely completion behind a longer-but-higher-average one.
"""

from __future__ import annotations


def beam_search(step_scorer, max_len: int, beam_width: int) -> list[int]:
    """Return the highest-scoring full path of length `max_len`.

    step_scorer(partial_path) -> list[float] of log-probabilities for the next
    token, given the path so far. Greedy takes argmax every step; beam keeps the
    top `beam_width` partial paths alive each step, which is what lets it cross
    a local low-probability "splitting point" into a much better continuation.
    """
    raise NotImplementedError("L06b")


def normalize_length(scores: list[float], alpha: float = 1.0) -> float:
    """Length-normalized path score: sum(scores) / len**alpha.

    A raw sum of log-probabilities is monotonically non-increasing, so without
    this the shortest completion always wins. alpha tunes how hard length is
    penalized; alpha=1 divides by length, the common default.
    """
    raise NotImplementedError("L06b")

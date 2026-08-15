"""Lecture 06b -- search-based decoding.

Two things are worth testing, and they map to the two halves of the lecture:

1. The "splitting point": greedy takes the per-step argmax and walks straight
   past a low-probability first token that leads to a much better continuation.
   Beam search keeps that path alive and finds it. This is why search exists at
   all -- and why it is not the same thing as greedy.
2. Length normalization: raw probability is monotonically non-increasing, so the
   shorter completion always wins. Dividing by length**alpha fixes it.
"""

from __future__ import annotations


def _splitting_scorer(path: list[int]) -> list[float]:
    """The canonical greedy-vs-beam grid.

    Step 0: token 0 (logp 0.6) beats token 1 (0.4), so greedy picks 0.
    But 0 dead-ends (best continuation 0.9), while 1 leads to 9.0.
    Beam width 2 keeps 1 alive and finds the 9.4 path greedy misses.
    """
    if not path:
        return [0.6, 0.4]
    if path[-1] == 0:
        return [0.1, 0.9]
    return [9.0, 8.0]


def _reference_beam(step_scorer, max_len: int, beam_width: int) -> list[int]:
    """Local reference beam search, used to validate the concept independent of
    your implementation."""
    best = [([], 0.0)]
    for _ in range(max_len):
        nxt: list[tuple[list[int], float]] = []
        for path, score in best:
            for token, lp in enumerate(step_scorer(path)):
                nxt.append((path + [token], score + lp))
        nxt.sort(key=lambda x: -x[1])
        best = nxt[:beam_width]
    return best[0][0]


def test_greedy_walks_past_the_splitting_point():
    """Greedy picks 0 then 1: 0.6 + 0.9 = 1.5. The best path is 1 then 0 = 9.4."""
    path = [0, 1]
    score = _splitting_scorer([])[0] + _splitting_scorer([0])[1]
    assert score == 1.5
    assert score < 9.4, "greedy's path is the trap; the good path scores 9.4"


def test_beam_finds_the_better_continuation():
    best = _reference_beam(_splitting_scorer, max_len=2, beam_width=2)
    assert best == [1, 0], f"expected [1,0] (score 9.4), got {best}"


def test_beam_matches_your_implementation():
    from engine.beam_search import beam_search

    assert beam_search(_splitting_scorer, max_len=2, beam_width=2) == [1, 0]


def test_shorter_completion_wins_without_normalization():
    """Raw sum of log-probs is non-increasing, so the 1-token path 'wins'."""
    short = [-2.0]                    # one token
    long = [-1.0, -1.0, -1.0, -1.0, -1.0]  # five tokens, higher average
    assert sum(short) > sum(long), "raw score prefers the short completion"


def test_length_normalization_flips_it():
    from engine.beam_search import normalize_length

    short = [-2.0]
    long = [-1.0, -1.0, -1.0, -1.0, -1.0]
    assert normalize_length(short) == -2.0          # -2.0 / 1
    assert normalize_length(long) == -1.0           # -5.0 / 5
    assert normalize_length(long) > normalize_length(short), (
        "normalized, the longer higher-average path wins"
    )


def test_normalization_alpha_tunes_length_penalty():
    from engine.beam_search import normalize_length

    # alpha > 1 penalizes length harder; alpha < 1 relaxes it.
    long = [-1.0, -1.0, -1.0, -1.0, -1.0]
    assert normalize_length(long, alpha=2.0) == -5.0 / 5 ** 2.0
    assert normalize_length(long, alpha=0.0) == -5.0

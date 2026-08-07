"""Lecture 03 -- naive generation.

The gate: your greedy loop must reproduce HuggingFace's greedy output EXACTLY.

Greedy decoding is deterministic, so this isn't a fuzzy "close enough" check --
any divergence is a real bug. Getting this green now is what lets you optimize
hard later: from Lecture 05 onward, every faster path is checked against the
same reference, so you find out immediately when speed cost you correctness.

These fail until you implement engine/model.py::load and
engine/generate.py::generate_naive. That's expected -- red is the starting state.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_naive_matches_huggingface_greedy(model_and_tokenizer, reference_greedy):
    """The correctness gate for the whole course."""
    from engine.generate import generate_naive

    model, tok = model_and_tokenizer
    text = generate_naive(model, tok, reference_greedy["prompt"], max_tokens=24)

    assert text == reference_greedy["text"], (
        "Greedy output diverged from HuggingFace.\n"
        f"  expected: {reference_greedy['text']!r}\n"
        f"  got:      {text!r}\n"
        "Greedy is deterministic -- this is a bug, not sampling noise. Common "
        "causes: sampling the wrong logit position (want [:, -1]), dropping the "
        "prompt from the output, or an off-by-one in the loop."
    )


@pytest.mark.slow
def test_respects_max_tokens(model_and_tokenizer):
    from engine.generate import generate_naive

    model, tok = model_and_tokenizer
    prompt = "Counting: one two three"
    n_prompt = len(tok(prompt).input_ids)

    text = generate_naive(model, tok, prompt, max_tokens=8)
    n_total = len(tok(text).input_ids)
    assert n_total <= n_prompt + 8


@pytest.mark.slow
def test_on_token_fires_once_per_token(model_and_tokenizer):
    """The harness hooks generation through on_token -- it must be exact.

    If this over- or under-counts, every tok/s number in the course is wrong by
    the same factor and nothing else will reveal it.
    """
    from engine.generate import generate_naive

    model, tok = model_and_tokenizer
    calls = []
    generate_naive(model, tok, "Hello", max_tokens=6, on_token=calls.append)
    assert len(calls) == 6


@pytest.mark.slow
def test_deterministic_across_runs(model_and_tokenizer):
    """Greedy has no randomness. Two runs must be identical."""
    from engine.generate import generate_naive

    model, tok = model_and_tokenizer
    a = generate_naive(model, tok, "The bottleneck is", max_tokens=12)
    b = generate_naive(model, tok, "The bottleneck is", max_tokens=12)
    assert a == b

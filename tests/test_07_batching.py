"""Lecture 07 -- static batching.

The gate: batching must not change what the model says. A batched greedy run
must produce exactly what each sequence produces alone.

When this fails, the cause is almost always padding leaking into the result --
either right-padding (so logits[:, -1] reads a PAD position) or an attention
mask that doesn't cover the pads.
"""

from __future__ import annotations

import pytest

PROMPTS = [
    "The KV cache exists because",
    "Hi",
    "Write a function that computes the arithmetic intensity of",
]


@pytest.mark.slow
def test_batched_matches_individual(model_and_tokenizer):
    """Batch of 3 with very different lengths -- padding is exercised hard."""
    from engine.generate import generate_batched, generate_cached

    model, tok = model_and_tokenizer

    individually = [generate_cached(model, tok, p, max_tokens=16) for p in PROMPTS]
    batched = generate_batched(model, tok, PROMPTS, max_tokens=16)

    assert len(batched) == len(PROMPTS)
    for i, (got, want) in enumerate(zip(batched, individually)):
        assert got == want, (
            f"Sequence {i} (prompt {PROMPTS[i]!r}) diverged when batched.\n"
            f"  alone:   {want!r}\n"
            f"  batched: {got!r}\n"
            "Check padding_side='left' and that the attention mask covers pads."
        )


@pytest.mark.slow
def test_single_item_batch_matches_unbatched(model_and_tokenizer):
    """Degenerate case, and a good first thing to fix when the above fails."""
    from engine.generate import generate_batched, generate_cached

    model, tok = model_and_tokenizer
    alone = generate_cached(model, tok, PROMPTS[0], max_tokens=12)
    batched = generate_batched(model, tok, [PROMPTS[0]], max_tokens=12)
    assert batched[0] == alone


@pytest.mark.slow
def test_order_is_preserved(model_and_tokenizer):
    """Results must come back in request order, not completion order."""
    from engine.generate import generate_batched, generate_cached

    model, tok = model_and_tokenizer
    reversed_prompts = list(reversed(PROMPTS))
    batched = generate_batched(model, tok, reversed_prompts, max_tokens=12)
    expected_first = generate_cached(model, tok, reversed_prompts[0], max_tokens=12)
    assert batched[0] == expected_first


class TestWasteAccounting:
    """Pure arithmetic -- the numbers behind the lecture. No model needed."""

    def test_uniform_load_wastes_nothing(self):
        from book.code.batching_waste import static_batch_cost

        reqs = [(64, 128)] * 16  # identical requests
        s = static_batch_cost(reqs, batch_size=8)
        assert s["decode_slots"] == s["decode_useful"]

    def test_mixed_load_wastes_a_lot(self):
        """One long request holds seven short slots hostage."""
        from book.code.batching_waste import static_batch_cost

        reqs = [(16, 8)] * 7 + [(16, 512)]
        s = static_batch_cost(reqs, batch_size=8)
        waste = 1 - s["decode_useful"] / s["decode_slots"]
        assert waste > 0.8, f"expected heavy waste, got {waste:.1%}"

    def test_bigger_batches_waste_more(self):
        """The trap: utilization and waste rise together under static batching."""
        from book.code.batching_waste import static_batch_cost
        from bench.workloads import mixed_length

        reqs = [(len(r.prompt.split()), r.max_tokens)
                for r in mixed_length(n=32, seed=0).requests]
        wastes = []
        for bs in (2, 4, 8, 16):
            s = static_batch_cost(reqs, bs)
            wastes.append(1 - s["decode_useful"] / s["decode_slots"])
        assert wastes == sorted(wastes)

    def test_continuous_batching_has_no_slot_waste(self):
        """The target Lecture 08 aims at."""
        from book.code.batching_waste import continuous_batch_cost
        from bench.workloads import mixed_length

        reqs = [(len(r.prompt.split()), r.max_tokens)
                for r in mixed_length(n=32, seed=0).requests]
        c = continuous_batch_cost(reqs)
        assert c["decode_slots"] == c["decode_useful"]

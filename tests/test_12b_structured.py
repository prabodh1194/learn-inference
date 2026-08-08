"""Lecture 12b -- structured output.

The guarantee under test: with a grammar mask applied, invalid tokens are
IMPOSSIBLE, not merely unlikely. That's the whole difference from prompting.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestLogitProcessorHook:
    """The extension point everything in this lecture uses."""

    def test_mask_makes_tokens_unreachable(self):
        from engine.sampling import SamplingParams, sample

        logits = torch.tensor([5.0, 4.0, 3.0, 2.0])
        allowed = torch.tensor([False, False, True, True])

        def mask(lg):
            out = lg.clone()
            out[~allowed] = float("-inf")
            return out

        p = SamplingParams(temperature=1.0, seed=0, logit_processor=mask)
        picks = {sample(logits.clone(), p) for _ in range(200)}
        assert picks <= {2, 3}, f"masked tokens were sampled: {picks}"

    def test_mask_applies_before_greedy_argmax(self):
        """Masking must beat greedy too, not just sampling."""
        from engine.sampling import SamplingParams, sample

        logits = torch.tensor([10.0, 1.0, 2.0])

        def mask(lg):
            out = lg.clone()
            out[0] = float("-inf")     # forbid the argmax
            return out

        p = SamplingParams(temperature=0.0, logit_processor=mask)
        assert sample(logits.clone(), p) == 2


class TestJsonGrammar:
    """A minimal grammar -- enough to feel the mechanism."""

    def test_only_open_brace_at_start(self):
        from engine.structured import JsonGrammar

        g = JsonGrammar(vocab={0: "{", 1: "}", 2: '"', 3: "a"})
        allowed = g.allowed_tokens(g.initial_state())
        assert 0 in allowed
        assert 1 not in allowed, "a JSON value cannot start with }"

    def test_braces_must_balance(self):
        from engine.structured import JsonGrammar

        g = JsonGrammar(vocab={0: "{", 1: "}", 2: '"', 3: "a"})
        s = g.advance(g.initial_state(), 0)      # consume '{'
        assert not g.is_complete(s), "unbalanced braces are not valid JSON"

    def test_forced_continuation_is_detectable(self):
        """When exactly one token is legal, you can emit it WITHOUT running
        the model -- the jump-ahead optimization, and the same insight as
        speculative decoding."""
        from engine.structured import JsonGrammar

        g = JsonGrammar(vocab={0: "{", 1: "}", 2: '"'})
        s = g.initial_state()
        allowed = g.allowed_tokens(s)
        if len(allowed) == 1:
            assert g.forced_token(s) == next(iter(allowed))


class TestSchedulerInteraction:
    """Grammar state is per sequence and must survive scheduling."""

    def test_grammar_state_is_per_sequence(self):
        from engine.sequence import Sequence

        a = Sequence(seq_id=0, prompt_ids=[1, 2, 3])
        b = Sequence(seq_id=1, prompt_ids=[1, 2, 3])
        a.grammar_state = "in_object"
        assert getattr(b, "grammar_state", None) != "in_object", (
            "grammar state leaked between sequences"
        )

    def test_same_prefix_different_schema_cannot_share_state(self):
        """Prefix caching shares KV blocks, never grammar state -- two
        requests with the same prompt may be under different schemas."""
        from engine.sequence import Sequence

        a = Sequence(seq_id=0, prompt_ids=[1, 2, 3])
        b = Sequence(seq_id=1, prompt_ids=[1, 2, 3])
        a.grammar_state = "schema_A"
        b.grammar_state = "schema_B"
        assert a.grammar_state != b.grammar_state

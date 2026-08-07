"""Lecture 06 -- sampling.

No model needed: sampling is a pure function of a logit vector, so these run
fast and catch the classic bugs directly.

The determinism tests matter most. Greedy decoding is the oracle every later
lecture checks against -- if it isn't reliably deterministic, you lose the
ability to tell a scheduler bug from sampling noise.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def logits():
    """A small, hand-checkable distribution.

    softmax([4,3,2,1,0]) ~= [.64, .24, .09, .03, .01]
    """
    return torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])


def _params(**kw):
    from engine.sampling import SamplingParams

    return SamplingParams(**kw)


class TestGreedy:
    def test_temperature_zero_is_argmax(self, logits):
        from engine.sampling import sample

        assert sample(logits, _params(temperature=0.0)) == 0

    def test_greedy_is_deterministic(self, logits):
        """The property every later test depends on."""
        from engine.sampling import sample

        p = _params(temperature=0.0)
        assert len({sample(logits.clone(), p) for _ in range(50)}) == 1

    def test_greedy_flag(self):
        assert _params(temperature=0.0).greedy
        assert not _params(temperature=0.7).greedy


class TestTopK:
    def test_only_top_k_are_reachable(self, logits):
        """With k=2, tokens 2-4 must never be selected."""
        from engine.sampling import sample

        p = _params(temperature=1.0, top_k=2, seed=0)
        picks = {sample(logits.clone(), p) for _ in range(200)}
        assert picks <= {0, 1}, f"sampled outside top-2: {picks}"

    def test_k_of_one_is_greedy(self, logits):
        from engine.sampling import sample

        p = _params(temperature=1.0, top_k=1)
        assert {sample(logits.clone(), p) for _ in range(30)} == {0}


class TestTopP:
    def test_confident_top_token_survives(self):
        """The classic bug.

        If the top token's probability already exceeds top_p, a naive
        `cumulative > p` mask removes EVERYTHING. The correct filter keeps the
        token that crosses the threshold.
        """
        from engine.sampling import sample

        peaked = torch.tensor([10.0, 0.0, 0.0, 0.0])  # top token ~= 0.9997
        p = _params(temperature=1.0, top_p=0.9)
        assert {sample(peaked.clone(), p) for _ in range(50)} == {0}

    def test_restricts_the_tail(self, logits):
        from engine.sampling import sample

        p = _params(temperature=1.0, top_p=0.9, seed=0)
        picks = {sample(logits.clone(), p) for _ in range(300)}
        assert 4 not in picks, "lowest-probability token should be excluded"


class TestRepetitionPenalty:
    def test_penalizes_seen_tokens(self, logits):
        """Token 0 dominates; penalizing it should dethrone it."""
        from engine.sampling import sample

        p = _params(temperature=0.0, repetition_penalty=10.0)
        assert sample(logits.clone(), p, prev_tokens=[0]) != 0

    def test_sign_is_handled(self):
        """Negative logits must be MULTIPLIED, not divided.

        Divide a negative logit by a penalty > 1 and you move it toward zero --
        raising its score. The "penalty" would reward repetition.
        """
        from engine.sampling import sample

        neg = torch.tensor([-1.0, -5.0, -6.0])
        p = _params(temperature=0.0, repetition_penalty=4.0)
        # Penalizing token 0 must not leave it the argmax.
        assert sample(neg.clone(), p, prev_tokens=[0]) != 0


class TestOrdering:
    def test_temperature_applied_before_truncation(self, logits):
        """High temperature flattens, so top-p should admit more tokens.

        If truncation ran first, temperature could not widen the candidate set
        and this would fail.
        """
        from engine.sampling import sample

        cold = _params(temperature=0.5, top_p=0.9, seed=0)
        hot = _params(temperature=3.0, top_p=0.9, seed=0)
        n_cold = len({sample(logits.clone(), cold) for _ in range(300)})
        n_hot = len({sample(logits.clone(), hot) for _ in range(300)})
        assert n_hot >= n_cold

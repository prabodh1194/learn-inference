"""Lecture 12 -- speculative decoding.

Model-free: propose and verify are pure functions.

The two tests that matter most are the token-accounting ones. Speculative
decoding is only worth doing if it's EXACT, and the two ways to break that are
dropping the correction token on a rejection (which can make speculation slower
than not speculating) and missing the bonus token on full acceptance (which
silently costs you the best case).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _logits_for(tokens: list[int], vocab: int = 100):
    """Build logits whose argmax at position i is tokens[i]."""
    out = torch.full((len(tokens), vocab), -10.0)
    for i, t in enumerate(tokens):
        out[i, t] = 10.0
    return out


class TestNgramProposal:
    def test_no_proposal_without_enough_context(self):
        from engine.speculative import NgramSpeculator

        assert NgramSpeculator(n=3).propose([1, 2]) == []

    def test_finds_a_repeated_pattern(self):
        """[1,2,3] appeared before, followed by [4,5]. Propose those."""
        from engine.speculative import NgramSpeculator

        spec = NgramSpeculator(n=3, max_draft_tokens=2)
        assert spec.propose([1, 2, 3, 4, 5, 9, 1, 2, 3]) == [4, 5]

    def test_no_match_gives_no_draft(self):
        from engine.speculative import NgramSpeculator

        assert NgramSpeculator(n=3).propose([1, 2, 3, 4, 5, 6]) == []

    def test_respects_max_draft_tokens(self):
        from engine.speculative import NgramSpeculator

        spec = NgramSpeculator(n=2, max_draft_tokens=3)
        tokens = [1, 2, 7, 8, 9, 10, 11, 1, 2]
        assert len(spec.propose(tokens)) <= 3

    def test_prefers_the_most_recent_match(self):
        """Recency is the better predictor -- search backwards."""
        from engine.speculative import NgramSpeculator

        spec = NgramSpeculator(n=2, max_draft_tokens=1)
        # [1,2] appears twice: followed by 3 early, by 9 recently.
        assert spec.propose([1, 2, 3, 0, 0, 1, 2, 9, 0, 1, 2]) == [9]


class TestVerify:
    def test_all_accepted_yields_a_bonus_token(self):
        """N drafts verified -> N+1 tokens. The last logits are a real
        prediction and it's free -- missing it costs you the best case."""
        from engine.speculative import verify

        draft = [10, 11, 12]
        logits = _logits_for([10, 11, 12, 13])   # 4 positions for 3 drafts
        accepted, n = verify(logits, draft)
        assert accepted == [10, 11, 12, 13]
        assert n == 4

    def test_rejection_still_produces_a_token(self):
        """The classic bug.

        Draft[1] is wrong, but the model's own prediction there is correct by
        definition -- keep it. Drop it and a full rejection produces NOTHING,
        making speculation slower than plain decoding.
        """
        from engine.speculative import verify

        draft = [10, 11, 12]
        logits = _logits_for([10, 99, 50, 60])   # position 1 disagrees
        accepted, n = verify(logits, draft)
        assert accepted == [10, 99], f"expected the correction kept, got {accepted}"
        assert n == 2

    def test_first_token_rejected_still_advances(self):
        """Worst case must still make progress, or you can livelock."""
        from engine.speculative import verify

        accepted, n = verify(_logits_for([77, 0, 0]), [10, 11])
        assert accepted == [77]
        assert n == 1

    def test_nothing_after_a_rejection_is_kept(self):
        """Later drafts were conditioned on a token that turned out wrong."""
        from engine.speculative import verify

        draft = [10, 11, 12, 13]
        logits = _logits_for([10, 99, 12, 13, 14])
        accepted, _ = verify(logits, draft)
        assert accepted == [10, 99]
        assert 12 not in accepted[2:]

    def test_empty_draft_is_plain_decoding(self):
        from engine.speculative import verify

        accepted, n = verify(_logits_for([42]), [])
        assert accepted == [42] and n == 1


class TestAcceptanceAccounting:
    """The metric the lecture insists you report."""

    def test_acceptance_rate_of_a_full_hit(self):
        from engine.speculative import verify

        draft = [1, 2, 3]
        _, n = verify(_logits_for([1, 2, 3, 4]), draft)
        assert (n - 1) / len(draft) == 1.0

    def test_acceptance_rate_of_a_total_miss(self):
        from engine.speculative import verify

        draft = [1, 2, 3]
        _, n = verify(_logits_for([9, 9, 9, 9]), draft)
        assert (n - 1) / len(draft) == 0.0


class TestWorkloadContrast:
    """The shipped workloads must actually differ in predictability.

    Deliberately uses a local reference n-gram search rather than your
    NgramSpeculator, so this validates the WORKLOADS and passes before you
    implement anything. If these two workloads don't differ, the lecture's
    central contrast can't be demonstrated no matter how good your code is.
    """

    @staticmethod
    def _repeat_rate(text: str, n: int = 3) -> float:
        """Fraction of n-grams in the text that occur more than once."""
        words = text.split()
        if len(words) <= n:
            return 0.0
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        seen: dict[tuple, int] = {}
        for g in grams:
            seen[g] = seen.get(g, 0) + 1
        return sum(1 for g in grams if seen[g] > 1) / len(grams)

    def test_code_is_more_self_similar_than_prose(self):
        from bench.workloads import code_completion, prose

        code = max(self._repeat_rate(r.prompt)
                   for r in code_completion(n=8).requests)
        text = max(self._repeat_rate(r.prompt) for r in prose(n=8).requests)

        assert code > text, (
            f"code repetition {code:.2f} should exceed prose {text:.2f}; "
            "otherwise these workloads cannot show the acceptance-rate gap"
        )


@pytest.mark.slow
def test_speculative_matches_greedy(model_and_tokenizer):
    """Non-negotiable: speculation is exact, not approximate."""
    from engine.generate import generate_cached, generate_speculative

    model, tok = model_and_tokenizer
    prompt = "def compute_intensity(flops, bytes_moved):"
    assert generate_speculative(model, tok, prompt, max_tokens=24) == (
        generate_cached(model, tok, prompt, max_tokens=24)
    )

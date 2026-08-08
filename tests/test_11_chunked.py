"""Lecture 11 -- chunked prefill.

Scheduler logic, so no model needed.

The ordering test is the one that matters. If you admit new sequences before
finishing in-flight prefills, you accumulate half-prefilled sequences that hold
KV memory while producing nothing -- memory pressure with no output, and every
request's TTFT gets worse.
"""

from __future__ import annotations

import pytest


def _seq(seq_id: int, prompt_len: int = 2000, max_tokens: int = 16):
    from engine.sequence import Sequence

    return Sequence(seq_id=seq_id, prompt_ids=list(range(prompt_len)),
                    max_tokens=max_tokens)


def _sched(**kw):
    from engine.scheduler import Scheduler

    kw.setdefault("max_batch_size", 8)
    kw.setdefault("max_batched_tokens", 1024)
    kw.setdefault("chunked_prefill", True)
    kw.setdefault("chunk_size", 512)
    return Scheduler(**kw)


class TestSequenceState:
    """These pass today -- engine/sequence.py is already implemented."""

    def test_prefill_done_tracks_progress(self):
        seq = _seq(0, prompt_len=1000)
        assert not seq.is_prefill_done
        seq.num_prefilled = 512
        assert not seq.is_prefill_done
        seq.num_prefilled = 1000
        assert seq.is_prefill_done

    def test_len_is_total_context(self):
        seq = _seq(0, prompt_len=100)
        seq.output_ids.extend([1, 2, 3])
        assert len(seq) == 103


class TestChunking:
    def test_long_prefill_is_split(self):
        """A 2000-token prompt at chunk 512 must not land in one step."""
        s = _sched(chunk_size=512, max_batched_tokens=1024)
        seq = _seq(0, prompt_len=2000)
        s.add(seq)
        prefill, _ = s.schedule()
        assert prefill, "should have scheduled some prefill"
        assert seq.num_prefilled <= 512, (
            f"prefilled {seq.num_prefilled} in one step; chunk_size is 512"
        )

    def test_prefill_completes_across_steps(self):
        s = _sched(chunk_size=512)
        seq = _seq(0, prompt_len=2000)
        s.add(seq)
        for _ in range(20):
            s.schedule()
            if seq.is_prefill_done:
                break
        assert seq.is_prefill_done
        assert seq.num_prefilled == 2000, "must not over- or under-shoot"

    def test_no_chunking_when_disabled(self):
        s = _sched(chunked_prefill=False, max_batched_tokens=8192)
        seq = _seq(0, prompt_len=2000)
        s.add(seq)
        s.schedule()
        assert seq.num_prefilled == 2000, "unchunked prefill runs in one step"


class TestDecodePriority:
    def test_decode_tokens_are_budgeted_first(self):
        """Running users are protected; prefill fills what's left.

        Reverse this and a long prompt starves everyone already decoding.
        """
        s = _sched(chunk_size=512, max_batched_tokens=520, max_batch_size=8)
        running = [_seq(i, prompt_len=8) for i in range(1, 9)]
        for r in running:
            s.add(r)
        for _ in range(3):
            s.schedule()          # get them into decode

        s.add(_seq(99, prompt_len=4000))
        prefill, decode = s.schedule()
        assert len(decode) >= 1, "running sequences must keep decoding"
        chunk = sum(getattr(x, "chunk_size_this_step", 0) for x in prefill)
        assert chunk <= 520 - len(decode), "prefill exceeded the leftover budget"

    def test_in_flight_prefill_finishes_before_new_admission(self):
        """Otherwise half-prefilled sequences pile up holding KV memory."""
        s = _sched(chunk_size=256, max_batched_tokens=256, max_batch_size=8)
        first = _seq(0, prompt_len=2000)
        s.add(first)
        s.schedule()

        second = _seq(1, prompt_len=100)
        s.add(second)
        prefill, _ = s.schedule()

        assert first in prefill, "in-flight prefill should continue"
        assert second not in prefill, (
            "admitted a new sequence while one was mid-prefill and the budget "
            "was exhausted"
        )


class TestBoundary:
    def test_final_chunk_transitions_to_decode(self):
        """The last prefill chunk produces the first token -- an easy
        off-by-one that duplicates or drops it."""
        s = _sched(chunk_size=512)
        seq = _seq(0, prompt_len=1024)
        s.add(seq)
        for _ in range(10):
            prefill, decode = s.schedule()
            if seq.is_prefill_done:
                break
            for x in prefill:
                pass
        assert seq.is_prefill_done
        _, decode = s.schedule()
        assert seq in decode, "a finished prefill should move to decode"

    def test_exact_multiple_of_chunk_size(self):
        s = _sched(chunk_size=512)
        seq = _seq(0, prompt_len=1024)     # exactly 2 chunks
        s.add(seq)
        steps = 0
        while not seq.is_prefill_done and steps < 10:
            s.schedule()
            steps += 1
        assert seq.num_prefilled == 1024
        assert steps <= 3, f"1024 tokens at chunk 512 took {steps} steps"


@pytest.mark.parametrize("prompt_len,chunk", [(100, 512), (512, 512), (513, 512)])
def test_short_prompts_are_not_over_chunked(prompt_len, chunk):
    """A prompt shorter than chunk_size needs exactly one step."""
    s = _sched(chunk_size=chunk)
    seq = _seq(0, prompt_len=prompt_len)
    s.add(seq)
    s.schedule()
    assert seq.num_prefilled == min(prompt_len, chunk)

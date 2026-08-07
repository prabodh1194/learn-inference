"""Lecture 08 -- continuous batching.

Scheduling is the hardest logic in Part II and it needs no model, so these
tests run in milliseconds. Iterate here, not against a GPU.

The behaviours pinned below are the ones that separate a real scheduler from a
loop: retire-before-admit, FIFO fairness, and budget accounting that counts
prefill tokens rather than sequences.
"""

from __future__ import annotations

import pytest


def _seq(seq_id: int, prompt_len: int = 8, max_tokens: int = 16):
    from engine.sequence import Sequence

    return Sequence(
        seq_id=seq_id,
        prompt_ids=list(range(prompt_len)),
        output_ids=[],
        max_tokens=max_tokens,
    )


def _sched(**kw):
    from engine.scheduler import Scheduler

    kw.setdefault("max_batch_size", 4)
    kw.setdefault("max_batched_tokens", 1024)
    return Scheduler(**kw)


class TestAdmission:
    def test_admits_up_to_batch_size(self):
        s = _sched(max_batch_size=4)
        for i in range(10):
            s.add(_seq(i))
        prefill, decode = s.schedule()
        assert len(prefill) == 4
        assert decode == []

    def test_leftovers_stay_queued(self):
        s = _sched(max_batch_size=2)
        for i in range(5):
            s.add(_seq(i))
        s.schedule()
        assert len(s.waiting) == 3

    def test_token_budget_limits_admission(self):
        """A step's cost is TOKENS, not sequences.

        Four 4000-token prefills is vastly more work than four decode steps.
        A scheduler that only counts sequences will build steps it can't afford.
        """
        s = _sched(max_batch_size=8, max_batched_tokens=100)
        for i in range(8):
            s.add(_seq(i, prompt_len=40))
        prefill, _ = s.schedule()
        assert len(prefill) <= 2, "40-token prompts, 100-token budget -> 2 max"


class TestRunning:
    def test_running_sequences_decode_each_step(self):
        s = _sched(max_batch_size=4)
        for i in range(2):
            s.add(_seq(i))
        s.schedule()                      # admitted -> prefill
        prefill, decode = s.schedule()    # next step -> decode
        assert prefill == []
        assert len(decode) == 2

    def test_finished_sequences_are_retired(self):
        from engine.sequence import Status

        s = _sched(max_batch_size=4)
        seq = _seq(0, max_tokens=1)
        s.add(seq)
        s.schedule()
        seq.output_ids.append(999)        # hit max_tokens
        s.schedule()
        assert seq.status == Status.FINISHED
        assert seq not in s.running

    def test_retire_frees_a_slot_in_the_same_step(self):
        """Retire-before-admit. Ordering here costs a step of latency per
        admission if you get it backwards."""
        s = _sched(max_batch_size=2)
        a, b, c = _seq(0, max_tokens=1), _seq(1), _seq(2)
        for x in (a, b, c):
            s.add(x)

        s.schedule()                      # a and b admitted, c waits
        a.output_ids.append(999)          # a is done
        prefill, _ = s.schedule()
        assert c in prefill, "freed slot should be refilled immediately"


class TestFairness:
    def test_fifo_no_head_of_line_jumping(self):
        """When the queue head doesn't fit, STOP -- don't skip past it.

        Admitting smaller requests behind a large one starves the large one
        indefinitely under sustained load.
        """
        s = _sched(max_batch_size=8, max_batched_tokens=50)
        big = _seq(0, prompt_len=100)
        small = _seq(1, prompt_len=5)
        s.add(big)
        s.add(small)

        prefill, _ = s.schedule()
        assert small not in prefill, (
            "small request jumped the queue -- use break, not continue"
        )

    def test_order_preserved(self):
        s = _sched(max_batch_size=3)
        for i in range(3):
            s.add(_seq(i))
        prefill, _ = s.schedule()
        assert [x.seq_id for x in prefill] == [0, 1, 2]


class TestNoWaste:
    def test_all_sequences_eventually_complete(self):
        """Drive the scheduler to completion; nothing may be lost or stuck."""
        s = _sched(max_batch_size=4)
        seqs = [_seq(i, max_tokens=(i % 5) + 1) for i in range(12)]
        for x in seqs:
            s.add(x)

        for _ in range(500):
            prefill, decode = s.schedule()
            if not prefill and not decode and not s.waiting:
                break
            for x in prefill + decode:
                x.output_ids.append(1)

        from engine.sequence import Status

        assert all(x.status == Status.FINISHED for x in seqs)

    def test_mixed_lengths_do_not_block_each_other(self):
        """The whole point: one long request must not hold short ones hostage."""
        s = _sched(max_batch_size=4)
        long_seq = _seq(0, max_tokens=100)
        shorts = [_seq(i, max_tokens=2) for i in range(1, 4)]
        s.add(long_seq)
        for x in shorts:
            s.add(x)

        steps = 0
        from engine.sequence import Status

        while steps < 50 and not all(x.status == Status.FINISHED for x in shorts):
            prefill, decode = s.schedule()
            for x in prefill + decode:
                x.output_ids.append(1)
            steps += 1

        assert all(x.status == Status.FINISHED for x in shorts)
        assert long_seq.status != Status.FINISHED, "long one should still run"
        assert steps < 10, f"shorts should finish fast, took {steps} steps"

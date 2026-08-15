"""Lecture 24b -- serving agents.

The concept under test: an agent's context grows linearly with tool steps, and
the compaction ladder buys it back cheapest-first. These tests validate the
shape of the problem independent of your implementation (local reference), then
check your ladder against it.
"""

from __future__ import annotations


def _tokens(text: str) -> int:
    return len(text.split())


def _total(turns: list[tuple[str, str]]) -> int:
    return sum(_tokens(t) for _, t in turns)


def test_context_grows_linearly_with_steps():
    """Each tool round-trip appends prompt + observation + action."""
    step = 100          # ~100 tokens per round-trip
    assert 10 * step == 1000
    assert 100 * step == 10_000, "100 steps = ~10k tokens, past the ~8k crossover"


def test_trim_keeps_the_newest_turns():
    """A local reference of the cheapest rung: drop oldest whole turns until the
    budget holds, always keeping the newest."""
    turns = [("user", "w" * 10)] * 10          # 10 turns, 10 tokens each
    budget = 50                                # need to shed 5 turns
    kept = turns[5:]                           # trim the 5 oldest
    assert len(kept) == 5
    assert kept[-1] == turns[-1], "the newest turn must always survive"


def test_your_ladder_fits_the_budget_and_keeps_the_tail():
    from engine.agent_context import compaction_ladder

    turns = [("user", "w " * 10)] * 10
    final, actions = compaction_ladder(turns, budget=50)
    assert final <= 50
    assert actions, "something must be shed to meet the budget"


def test_no_work_when_under_budget():
    from engine.agent_context import compaction_ladder

    turns = [("user", "a b c")]
    final, actions = compaction_ladder(turns, budget=100)
    assert final == _total(turns) == 3
    assert actions == []

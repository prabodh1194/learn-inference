"""L24b -- serving agents.

An agent is a loop: model call -> tool call -> observation -> repeat. Every
round-trip appends prompt + observation + action, so context grows linearly
with the number of steps and the KV cache (L05) is the binding constraint.
Production harnesses manage it with a lazy escalation ladder, cheapest move
first: trim -> snip -> micro-compact -> collapse -> auto-compact.

The number that must move: tokens in context after N tool steps, before and
after the ladder.
"""

from __future__ import annotations


def compaction_ladder(turns: list[tuple[str, str]], budget: int):
    """Cheapest-first context reduction until the transcript fits `budget`.

    turns: list of (role, text) from oldest to newest. Roles are "system",
    "user", "assistant", "tool". Return (final_token_count, actions_taken).

    actions_taken is a list of ("trim"|"snip"|"compact", n_tokens_freed) in the
    order applied, newest turn always kept. "trim" drops the oldest turn whole;
    "snip" truncates a turn's text; "compact" replaces several turns with one
    summary of the same token count. Only climb the ladder as far as needed.
    """
    raise NotImplementedError("L24b")

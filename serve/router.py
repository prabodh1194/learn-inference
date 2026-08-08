"""M4.4 (Lecture 27) -- routing across replicas.

Round-robin balances LOAD and ignores STATE. The KV cache is state, so on
multi-turn traffic round-robin sends a user's follow-up to a replica that has
never seen their conversation -- and Lecture 10's win evaporates exactly when
you scale.

The tension is unavoidable: the replica holding your cache may be the busy
one. Every production router blends affinity with load, and the blend is a
tuning decision rather than a solved problem.
"""

from __future__ import annotations


class Router:
    """M4.4. Strategies: round_robin, session, cache_aware."""

    def __init__(self, n_replicas: int, strategy: str = "round_robin",
                 load_threshold: float = 0.9):
        raise NotImplementedError("M4.4")

    def route(self, prefix_hash: str) -> int:
        """Pick a replica. cache_aware must fall back when the warm one is
        overloaded, or you build a hotspot."""
        raise NotImplementedError("M4.4")

    def mark_warm(self, replica: int, prefix_hash: str) -> None:
        raise NotImplementedError("M4.4")

    def set_load(self, replica: int, load: float) -> None:
        raise NotImplementedError("M4.4")


def simulate_multiturn(router: "Router", n_conversations: int = 50,
                       turns: int = 5) -> dict:
    """M4.4. Replay multi-turn traffic; report {"hit_rate": float}.

    This is where cache-aware routing should beat round-robin decisively.
    """
    raise NotImplementedError("M4.4")

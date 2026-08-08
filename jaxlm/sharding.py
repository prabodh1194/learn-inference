"""M3.3 (Lecture 22) -- declarative tensor parallelism.

You annotate a LAYOUT; XLA derives that a row-parallel matmul needs an
all-reduce and inserts it. Dump the HLO and find the collective you never
wrote -- that is the moment this lecture is built around.

Then port it to PyTorch by hand (engine/parallel.py), placing the same
collective in the same position.
"""

from __future__ import annotations


def make_mesh(axis_name: str = "tp"):
    """M3.3. Device mesh over the available accelerators."""
    raise NotImplementedError("M3.3")


def shard_params(params, mesh):
    """M3.3. NamedSharding: P(None, "tp") for column, P("tp", None) for row."""
    raise NotImplementedError("M3.3")


def find_collectives(lowered) -> list[str]:
    """M3.3. Extract all-reduce / all-gather ops from the HLO.

    Seeing the collective XLA inserted, at exactly the position the math
    requires, is the payoff.
    """
    raise NotImplementedError("M3.3")

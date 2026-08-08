"""M1.10 (Lecture 12b) -- grammar-constrained decoding.

Mask logits so invalid tokens are IMPOSSIBLE rather than unlikely. That
guarantee is the whole difference from asking politely in the prompt.

The cost is CPU work per step per sequence over a 151,936-token vocabulary --
squarely the kind of thing that stalls the engine loop (Lecture 24). Real
implementations precompile the FSM, cache it per schema, and use token tries.
"""

from __future__ import annotations


class JsonGrammar:
    """A minimal JSON state machine. Enough to feel the mechanism.

    Not a schema engine -- for that, see XGrammar or Outlines.
    """

    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab

    def initial_state(self):
        raise NotImplementedError("M1.10")

    def allowed_tokens(self, state) -> set[int]:
        """Token ids legal in this state. Everything else gets -inf."""
        raise NotImplementedError("M1.10")

    def advance(self, state, token_id: int):
        raise NotImplementedError("M1.10")

    def is_complete(self, state) -> bool:
        """True only when the output is a complete, valid document."""
        raise NotImplementedError("M1.10")

    def forced_token(self, state) -> int | None:
        """When exactly one token is legal, return it.

        Jump-ahead: emit it WITHOUT a forward pass. Same insight as
        speculative decoding, reached from a different direction.
        """
        allowed = self.allowed_tokens(state)
        return next(iter(allowed)) if len(allowed) == 1 else None

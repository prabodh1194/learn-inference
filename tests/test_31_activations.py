"""Lecture 31 -- Activations, checkpointing, and the output-head logit trap."""

from __future__ import annotations

from book.code.activation_budget import L, layer_bytes
from train.checkpointing import (
    checkpointed_activation_bytes,
    layer_activation_bytes,
    logit_bytes,
)


class TestBookArithmetic:
    def test_flash_cuts_activations_7_7x_at_4k(self):
        eager = L * layer_bytes(s=4096, b=4, flash=False)
        flash = L * layer_bytes(s=4096, b=4, flash=True)
        assert round(eager / (1024**3), 2) == 128.62
        assert round(flash / (1024**3), 2) == 16.65

    def test_unfused_fp32_logits_exceed_9_gib(self):
        logits_gib = (4 * 4096 * 151_936 * 4) / (1024**3)
        assert round(logits_gib, 2) == 9.27


class TestImplementation:
    def test_layer_activation_bytes_flash_vs_eager(self):
        f_bytes = layer_activation_bytes(seq_len=4096, batch=4, flash=True)
        e_bytes = layer_activation_bytes(seq_len=4096, batch=4, flash=False)
        assert e_bytes > f_bytes * 7

    def test_checkpointed_activation_bytes(self):
        got = checkpointed_activation_bytes(seq_len=4096, batch=4, n_layers=28, hidden=1024)
        assert got == 28 * 2 * 4096 * 4 * 1024

    def test_logit_bytes(self):
        assert logit_bytes(seq_len=4096, batch=4) == 4 * 4096 * 151_936 * 4

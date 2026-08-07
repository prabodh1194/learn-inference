"""Lecture 02 -- arithmetic intensity.

Pure arithmetic: no GPU, no model, no download. These pass today.

They pin the derivation against Kiely's published worked example (§2.4.2,
Figs 2.14-2.18, p.63-66). If a refactor ever breaks the math, the number stops
matching the book and you find out here rather than by drawing a wrong
conclusion three lectures later.
"""

from __future__ import annotations

import pytest

from book.code.roofline import (
    DEVICES,
    ModelDims,
    attention_intensity,
    decode_step_intensity,
    prefill_intensity,
)


class TestBookExample:
    """N=4096, d=128, FP16 -- the exact case worked in the book."""

    def test_reproduces_62_ops_byte(self):
        r = attention_intensity(4096, 128)
        assert r["intensity"] == pytest.approx(62, abs=1.0), (
            "Should match the book's ~62 ops:byte (Fig 2.18, p.66)"
        )

    def test_memory_movement_formula(self):
        """Book Fig 2.16: total = 8N^2 + 8Nd bytes."""
        n, d = 4096, 128
        assert attention_intensity(n, d)["memory_bytes"] == 8 * n * n + 8 * n * d

    def test_compute_formula(self):
        """Book Fig 2.17: total = 4(N^2)d + 3N^2 ops."""
        n, d = 4096, 128
        assert attention_intensity(n, d)["compute_ops"] == 4 * n * n * d + 3 * n * n


class TestBottleneckDirection:
    """The conclusion the whole course rests on."""

    def test_decode_is_memory_bound_on_every_device(self):
        """Decode regenerates one token per full pass over the weights.

        Robustness matters here: this must hold on an H100 (ridge 295) AND a
        3090 (ridge 76) AND an M1 (ridge 38). If it only held on one, it would
        be a hardware quirk rather than a property of decode.
        """
        intensity = decode_step_intensity(ModelDims(), seq_len=2048)
        assert intensity < 10, "decode intensity should be tiny"
        for dev in DEVICES:
            assert intensity < dev.ops_byte, f"{dev.name} should be memory-bound"

    def test_prefill_is_compute_bound_on_every_device(self):
        intensity = prefill_intensity(ModelDims(), n_tokens=2048)
        for dev in DEVICES:
            assert intensity > dev.ops_byte, f"{dev.name} should be compute-bound"

    def test_prefill_intensity_grows_with_batch(self):
        """Why batching helps: weights load once, amortized over more tokens.

        This is the same mechanism that makes continuous batching (L08) work --
        adding compute to a memory-bound phase is nearly free.
        """
        m = ModelDims()
        vals = [prefill_intensity(m, n) for n in (128, 512, 2048)]
        assert vals == sorted(vals)


class TestModelDims:
    def test_gqa_shrinks_the_kv_cache(self):
        """Qwen3-0.6B has 16 query heads but only 8 KV heads.

        The cache is sized by KV heads, not query heads -- so GQA halves it
        versus full multi-head attention. Decode is memory-bound, so this is a
        direct 2x on the thing that bottlenecks it.
        """
        gqa = ModelDims(n_heads=16, n_kv_heads=8)
        mha = ModelDims(n_heads=16, n_kv_heads=16)
        assert gqa.kv_bytes_per_token() * 2 == mha.kv_bytes_per_token()

    def test_kv_cache_scales_linearly_with_context(self):
        m = ModelDims()
        assert m.kv_bytes_per_token() * 4096 == pytest.approx(
            m.kv_bytes_per_token() * 2 * 2048
        )


class TestDevices:
    def test_ridge_matches_book_h100(self):
        """989 TFLOPS / 3.35 TB/s ~= 295 ops:byte (book §2.4.1, p.62)."""
        h100 = next(d for d in DEVICES if "H100" in d.name)
        assert h100.ops_byte == pytest.approx(295, abs=2)

    def test_3090_has_lower_ridge_than_h100(self):
        """More bandwidth per FLOP -> friendlier to memory-bound decode.

        Worth internalizing: a "slower" GPU can be relatively better at decode.
        """
        h100 = next(d for d in DEVICES if "H100" in d.name)
        rtx = next(d for d in DEVICES if "3090" in d.name)
        assert rtx.ops_byte < h100.ops_byte

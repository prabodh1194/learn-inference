"""Lecture 35 -- FP8 tile quantization and auxiliary-loss-free MoE balancing."""

from __future__ import annotations

import torch

from book.code.fp8_trainingsim import quant_dequant_e4m3
from train.fp8_train import tile_quantize_fp8, update_expert_biases


class TestBookArithmetic:
    def test_per_tensor_outlier_flushes_normal_values_to_zero(self):
        vals = [1.0] * 255 + [500_000.0]
        assert quant_dequant_e4m3(vals, block_size=256)[0] == 0.0
        assert quant_dequant_e4m3(vals, block_size=128)[0] == 1.0


class TestImplementation:
    def test_tile_quantize_fp8_isolates_outlier_tile(self):
        x = torch.ones(256)
        x[-1] = 500_000.0
        out = tile_quantize_fp8(x, tile_size=128)
        assert torch.allclose(out[:128], torch.ones(128))

    def test_update_expert_biases_nudges_by_gamma(self):
        got = update_expert_biases([0.0, 0.0, 0.0, 0.0], [64, 0, 0, 0], gamma=0.05)
        assert got == [-0.05, 0.05, 0.05, 0.05]

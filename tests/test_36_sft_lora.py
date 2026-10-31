"""Lecture 36 -- Supervised fine-tuning prompt masking and LoRA adapters."""

from __future__ import annotations

import torch

from book.code.lora_memory import qwen3_0_6b_lora_params
from train.sft_lora import build_sft_labels, lora_forward, lora_param_count


class TestBookArithmetic:
    def test_qwen_lora_r16_is_10_09m_params(self):
        assert qwen3_0_6b_lora_params(r=16) == 10_092_544


class TestImplementation:
    def test_build_sft_labels_masks_prompt_only(self):
        assert build_sft_labels([10, 11, 12], [20, 21]) == [-100, -100, -100, 20, 21]

    def test_lora_param_count(self):
        assert lora_param_count(d_in=1024, d_out=2048, rank=16) == 49_152

    def test_lora_zero_b_init_is_exact_identity(self):
        x = torch.randn(4, 16)
        w0 = torch.randn(32, 16)
        a = torch.randn(4, 16)
        b = torch.zeros(32, 4)
        assert torch.equal(lora_forward(x, w0, a, b, alpha=8.0), x @ w0.T)

"""M2.5 (Lecture 19) -- quantization arithmetic.

`quantized_bytes` is implemented (pure accounting). The quantize/dequantize
pair is yours: it is where per-tensor vs per-channel vs per-group actually
differs, and the test with an outlier channel shows why granularity matters.
"""

from __future__ import annotations


def quantized_bytes(n_params: int, bits: int,
                    group_size: int | None = 128) -> float:
    """Storage including scale/zero-point metadata.

    The metadata is why int4 with group_size=32 is not half of int8 -- finer
    groups mean more scales, and at some point the metadata dominates.
    """
    weight_bytes = n_params * bits / 8
    if group_size:
        n_groups = n_params / group_size
        weight_bytes += n_groups * 2 * 2      # fp16 scale + zero point
    return weight_bytes


def quantize_per_tensor(w, bits: int = 8):
    """M2.5. One scale for the whole tensor. Worst accuracy -- a single
    outlier stretches the range and crushes precision everywhere else."""
    raise NotImplementedError("M2.5")


def quantize_per_channel(w, bits: int = 8):
    """M2.5. One scale per output channel. The standard choice."""
    raise NotImplementedError("M2.5")


def dequantize(q, scale, zero_point):
    """M2.5. w ~= scale * (q - zero_point)."""
    raise NotImplementedError("M2.5")

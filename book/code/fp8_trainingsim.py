"""Lecture 35 demo -- FP8 per-tensor vs 1x128 tile scaling, and loss-free MoE balancing.

    uv run python book/code/fp8_trainingsim.py

Reproduces the outlier problem in FP8 E4M3 training and DeepSeek-V3's
1x128 tile quantization + auxiliary-loss-free bias thermostat.
"""

from __future__ import annotations

import math


def quant_dequant_e4m3(vals: list[float], block_size: int) -> list[float]:
    fp8_max, min_subnormal = 448.0, 2**-9
    out: list[float] = []
    for i in range(0, len(vals), block_size):
        tile = vals[i : i + block_size]
        amax = max(abs(v) for v in tile) or 1.0
        scale = amax / fp8_max
        for v in tile:
            scaled = v / scale
            if abs(scaled) < min_subnormal:
                q = 0.0
            else:
                exp = math.floor(math.log2(abs(scaled)))
                step = 2.0 ** (exp - 3)
                q = round(scaled / step) * step
            out.append(q * scale)
    return out


def print_fp8_outlier_demo() -> None:
    vals = [1.0] * 255 + [500_000.0]
    per_tensor = quant_dequant_e4m3(vals, block_size=256)
    per_tile = quant_dequant_e4m3(vals, block_size=128)
    print("FP8 E4M3 quantization of 255 values of 1.0 + one outlier of 500,000.0")
    print("=========================================================================")
    print(f"  per-tensor (1 scale for all 256) : tile-0 value 1.0 -> {per_tensor[0]:.4f}  (erased to zero!)")
    print(f"  1x128 tile (scale per 128 strip) : tile-0 value 1.0 -> {per_tile[0]:.4f}  (preserved exactly)\n")


def print_thermostat_demo() -> None:
    biases = [0.0, 0.0, 0.0, 0.0]
    raw_affinities = [0.80, 0.72, 0.65, 0.60]
    gamma = 0.05
    print("Auxiliary-loss-free MoE routing thermostat (4 experts, top-1, gamma=0.05)")
    print("=========================================================================")
    for step in range(1, 5):
        loads = [0, 0, 0, 0]
        winner = max(range(4), key=lambda i: raw_affinities[i] + biases[i])
        loads[winner] = 64
        target = sum(loads) / 4
        biases = [b - gamma if ld > target else b + gamma for b, ld in zip(biases, loads, strict=True)]
        print(f"  step {step}: routed to expert {winner} (gate={raw_affinities[winner]:.2f}, untainted) -> updated biases={[round(b, 2) for b in biases]}")


def main() -> None:
    print_fp8_outlier_demo()
    print_thermostat_demo()


if __name__ == "__main__":
    main()

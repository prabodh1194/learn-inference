from __future__ import annotations
import math


def chinchilla_loss(n_params: float, d_tokens: float) -> float:
    e_irr = 1.69
    a_coeff, alpha = 406.4, 0.34
    b_coeff, beta = 410.7, 0.28
    return e_irr + a_coeff / (n_params ** alpha) + b_coeff / (d_tokens ** beta)


def lifetime_flops(
    n_params: float, d_train: float, d_infer: float
) -> dict[str, float]:
    train_flops = 6.0 * n_params * d_train
    infer_flops = 2.0 * n_params * d_infer
    return {
        "train_flops": train_flops,
        "infer_flops": infer_flops,
        "total_flops": train_flops + infer_flops,
    }


def yarn_attn_scale(extension_factor: float) -> float:
    if extension_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(extension_factor) + 1.0


def _build_lifetime_lines(d_infer: float) -> list[str]:
    configs = [
        ("Chinchilla 2.8B (20 tok/param)", 2.8e9, 56.0e9),
        ("Over-trained 1.4B (300 tok/param)", 1.4e9, 420.0e9),
        ("Qwen3-0.6B scale (60,000 tok/param)", 0.6e9, 36.0e12),
    ]
    lines: list[str] = []
    for label, n_p, d_tr in configs:
        loss = 1.69 + 406.4 / (n_p ** 0.34) + 410.7 / (d_tr ** 0.28)
        tr_z = (6.0 * n_p * d_tr) / 1e21
        in_z = (2.0 * n_p * d_infer) / 1e21
        lines.append(
            f"{label:<36} | L={loss:.3f} | "
            f"train={tr_z:6.1f} ZFLOP | infer={in_z:6.1f} ZFLOP | "
            f"total={tr_z + in_z:6.1f} ZFLOP"
        )
    return lines


def _build_yarn_lines() -> list[str]:
    lines: list[str] = []
    for s_ext in (1, 4, 8, 32):
        scale = 1.0 if s_ext <= 1 else 0.1 * math.log(float(s_ext)) + 1.0
        lines.append(
            f"  s = {s_ext:>2}x -> sqrt(1/t) = {scale:.4f}, t = {1.0/(scale**2):.4f}"
        )
    return lines


def main() -> None:
    print("Over-training vs. Chinchilla (100T lifetime served tokens)")
    print("-" * 72)
    print("\n".join(_build_lifetime_lines(100.0e12)))
    print("\nYaRN attention temperature scale sqrt(1/t) = 0.1 * ln(s) + 1:")
    print("\n".join(_build_yarn_lines()))


if __name__ == "__main__":
    main()

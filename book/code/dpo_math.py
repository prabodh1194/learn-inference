"""Lecture 38 demo -- Direct Preference Optimization (DPO) mechanics.

    uv run python book/code/dpo_math.py

Tracks implicit rewards r_hat = beta * log(pi / pi_ref), the DPO loss, and the
per-example wrongness gradient weight sigmoid(r_hat(l) - r_hat(w)).
"""

from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def print_dpo_case(label: str, pi_w: float, pi_l: float, ref_w: float, ref_l: float, beta: float = 0.1) -> None:
    rw = beta * (pi_w - ref_w)
    rl = beta * (pi_l - ref_l)
    margin = rw - rl
    wt = sigmoid(rl - rw)
    loss = -math.log(sigmoid(margin))
    print(f"  {label:<32}  {rw:8.3f}   {rl:8.3f}  {margin:7.3f}    {wt:6.3f}  {loss:6.3f}")


def print_dpo_examples() -> None:
    print("DPO implicit rewards, loss, and gradient weight (beta = 0.1)")
    print("=========================================================================")
    print("  case                              r_hat(w)   r_hat(l)   margin   grad_wt   loss")
    print_dpo_case("Start of DPO (pi == pi_ref)", -12.0, -15.0, -12.0, -15.0)
    print_dpo_case("Policy ranks WRONG (prefers y_l)", -14.0, -11.0, -12.0, -15.0)
    print_dpo_case("Policy ranks RIGHT (prefers y_w)", -8.0, -20.0, -12.0, -15.0)


def main() -> None:
    print_dpo_examples()


if __name__ == "__main__":
    main()

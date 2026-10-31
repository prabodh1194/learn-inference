"""Lecture 37 demo -- the 4-model memory bill of PPO RLHF vs DPO vs GRPO.

    uv run python book/code/rlhf_memory.py

Compares resident model-state memory across SFT, PPO (Actor + Critic + Ref + RM),
DPO (Actor + Ref), and GRPO with verifiable rewards (Actor + Ref).
"""

from __future__ import annotations


def print_comparison(name: str, params: float) -> None:
    gib = 1024**3
    rows = [
        ("SFT / Pre-training", "Actor(16P)", 16 * params),
        ("PPO RLHF (4 models)", "Actor(16P) + Critic(16P) + Ref(2P) + RM(2P)", 36 * params),
        ("DPO (2 models)", "Actor(16P) + Ref(2P)", 18 * params),
        ("SimPO (1 model)", "Actor(16P) only", 16 * params),
        ("GRPO + RLVR (no critic/RM)", "Actor(16P) + Ref(2P)", 18 * params),
    ]
    print(f"Model-state VRAM bill for {name} ({params/1e9:.2f}B params)")
    print("=========================================================================")
    for method, formula, b in rows:
        print(f"  {method:<26} {formula:<43} {b/gib:7.2f} GiB")
    print()


def main() -> None:
    print_comparison("Qwen3-0.6B", 596_049_920)
    print_comparison("7B dense model", 7_000_000_000)


if __name__ == "__main__":
    main()

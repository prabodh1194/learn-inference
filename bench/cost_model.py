"""M4.6 (Lecture 28) -- dollars per million tokens.

The metric that unifies the whole book, and the one that reframes it:

    cost/M = (GPU $/hr) / (tokens/hr) * 1e6

At 20% utilization the same engine costs 5x what it does at 100%. Below
roughly 50% utilization, utilization dominates every kernel optimization in
Part III. Knowing which lever you're pulling is the skill.

Implemented -- it's arithmetic, and you should be able to run it today.
"""

from __future__ import annotations


def cost_per_million(dollars_per_hour: float, tokens_per_second: float,
                     utilization: float = 1.0) -> float:
    """Cost per million output tokens at a given average utilization."""
    if tokens_per_second <= 0 or utilization <= 0:
        return float("inf")
    tokens_per_hour = tokens_per_second * 3600 * utilization
    return dollars_per_hour / tokens_per_hour * 1_000_000


def compare_levers(dollars_per_hour: float, base_tps: float) -> None:
    """Show why utilization usually beats a kernel win."""
    print(f"\n{'scenario':<34}{'tok/s':>9}{'util':>8}{'$/M tok':>11}")
    print("-" * 62)
    rows = [
        ("baseline",                      base_tps,        0.50),
        ("30% faster kernels",            base_tps * 1.30, 0.50),
        ("2x batch (better packing)",     base_tps * 2.00, 0.50),
        ("utilization 50% -> 80%",        base_tps,        0.80),
        ("prefix caching (30% skipped)",  base_tps * 1.43, 0.50),
        ("idle at 20%",                   base_tps,        0.20),
    ]
    for label, tps, util in rows:
        print(f"{label:<34}{tps:>9.0f}{util:>8.0%}"
              f"{cost_per_million(dollars_per_hour, tps, util):>11.4f}")
    print("\nNote where the biggest swing is. It is rarely the kernels.")


def breakeven_utilization(dollars_per_hour: float, tokens_per_second: float,
                          target_cost_per_million: float) -> float:
    """Utilization needed to hit a target price. Often sobering."""
    if target_cost_per_million <= 0:
        return float("inf")
    tokens_needed = dollars_per_hour / target_cost_per_million * 1_000_000
    return tokens_needed / (tokens_per_second * 3600)


if __name__ == "__main__":
    compare_levers(dollars_per_hour=0.25, base_tps=2000)

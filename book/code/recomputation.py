"""Lecture 03 -- the cost of having no KV cache.

Runs anywhere: arithmetic only, no GPU, no model.

    uv run python book/code/recomputation.py

Counts the work a cacheless generation loop throws away. You are about to write
that loop by hand, so it helps to know the size of the hole before you fall in.
"""

from __future__ import annotations

from book.code.roofline import ModelDims


def kv_projections(d: ModelDims, n_tokens: int) -> int:
    """FLOPs to compute K and V for n_tokens.

    Two projections (K and V), each hidden -> n_kv_heads*head_dim, 2 FLOPs per
    multiply-accumulate.
    """
    kv_dim = d.n_kv_heads * d.head_dim
    return 2 * (2 * n_tokens * d.hidden * kv_dim) * d.n_layers


def main() -> None:
    d = ModelDims()

    print("Generating 512 tokens from a 64-token prompt, no cache")
    print("=" * 64)
    print("At every step the model re-runs the FULL sequence, so K and V for")
    print("token 0 get recomputed on every single step after it.\n")

    prompt, out = 64, 512
    print(f"{'step':>6}{'seq len':>10}{'K/V computed':>15}{'actually new':>14}"
          f"{'wasted':>9}")
    total_kv = wasted_kv = 0
    for step in (1, 2, 8, 64, 256, 512):
        seq = prompt + step - 1
        computed = seq
        # Step 1 is the first forward pass, so ALL of the prompt's K/V are
        # genuinely new. From step 2 on, only the newest token is unknown and
        # everything before it is a recomputation.
        new = seq if step == 1 else 1
        print(f"{step:>6}{seq:>10}{computed:>15}{new:>14}"
              f"{100*(computed-new)/computed:>8.1f}%")

    for step in range(1, out + 1):
        seq = prompt + step - 1
        total_kv += seq
        wasted_kv += 0 if step == 1 else seq - 1

    print(f"\nOver all {out} steps:")
    print(f"  K/V vectors computed : {total_kv:,}")
    print(f"  actually needed      : {out:,}")
    print(f"  thrown away          : {wasted_kv:,}  "
          f"({100*wasted_kv/total_kv:.1f}%)")

    flops_wasted = kv_projections(d, wasted_kv)
    print(f"\n  wasted compute       : {flops_wasted/1e12:.1f} TFLOP")
    print("  (projections only -- attention itself wastes more)")

    print("\n\nWhy the curve bends: work per step grows with position")
    print("=" * 64)
    print("Step n processes n tokens, so cost is O(n). Summed over N steps")
    print("that is O(N^2) -- the shape you are about to plot.\n")
    print(f"{'output len':>12}{'total K/V work':>18}{'vs 128 tokens':>16}")
    base = None
    for n in (128, 256, 512, 1024, 2048):
        work = sum(prompt + s - 1 for s in range(1, n + 1))
        base = base or work
        print(f"{n:>12}{work:>18,}{work/base:>15.1f}x")

    print("\nDouble the output length, roughly QUADRUPLE the work.")
    print("That is the curve you will measure in this lecture -- and flatten")
    print("in Lecture 05.")

    print("\n\nWhat the cache costs instead")
    print("=" * 64)
    kv_tok = d.kv_bytes_per_token()
    print(f"Storing K/V per token: {kv_tok/1024:.0f} KiB "
          f"({d.n_layers} layers x {d.n_kv_heads} KV heads x "
          f"{d.head_dim} dims x 2 (K,V) x {d.bytes_per_value}B)\n")
    print(f"{'seq len':>10}{'cache size':>14}")
    for n in (512, 2048, 8192, 32768):
        print(f"{n:>10}{kv_tok*n/1024**2:>11.0f} MiB")

    print("\nThe classic systems trade: spend memory, save compute.")
    print("Lectures 09 and 10 are then about spending that memory WELL.")


if __name__ == "__main__":
    main()

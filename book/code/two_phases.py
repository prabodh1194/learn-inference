"""Lecture 01 -- prefill and decode are different problems.

Runs on anything: pure arithmetic, no GPU, no model download.

    uv run python book/code/two_phases.py

Shows the asymmetry that the rest of the book is a response to. Prefill does
enormous work in one shot; decode does trivial work thousands of times, and
re-reads the entire model on every one of them.
"""

from __future__ import annotations

from book.code.roofline import ModelDims, model_params

# Verified against TechPowerUp; see roofline.py's Device docstring.
RTX_3090_BANDWIDTH = 936.2e9  # bytes/sec


def phase_work(d: ModelDims, prompt_tokens: int, output_tokens: int) -> dict:
    params = model_params(d)
    weight_bytes = params * d.bytes_per_value

    # PREFILL: every prompt token flows through the model together.
    # One pass over the weights, matrix-MATRIX multiplies.
    prefill_flops = 2 * params * prompt_tokens
    prefill_weight_reads = weight_bytes  # loaded once

    # DECODE: one token at a time, and every step re-reads every weight.
    # Matrix-VECTOR multiplies -- the same memory traffic for 1/N the work.
    decode_flops = 2 * params * output_tokens
    decode_weight_reads = weight_bytes * output_tokens  # <-- the whole problem

    # KV cache grows as we go; decode reads it back every step.
    kv_per_token = d.kv_bytes_per_token()
    avg_ctx = prompt_tokens + output_tokens / 2
    decode_kv_reads = kv_per_token * avg_ctx * output_tokens

    return {
        "prefill_flops": prefill_flops,
        "prefill_bytes": prefill_weight_reads,
        "decode_flops": decode_flops,
        "decode_bytes": decode_weight_reads + decode_kv_reads,
        "weight_bytes": weight_bytes,
        "params": params,
    }


def main() -> None:
    d = ModelDims()
    print(f"{d.name}: ~{model_params(d)/1e6:.0f}M params (approx), "
          f"{model_params(d) * d.bytes_per_value / 1024**2:.0f} MiB of weights\n")

    print("A typical chat request: 512-token prompt, 256 tokens out")
    print("=" * 62)
    w = phase_work(d, prompt_tokens=512, output_tokens=256)

    print(f"{'':<10}{'compute':>14}{'memory read':>16}{'ops:byte':>12}")
    for phase in ("prefill", "decode"):
        f, b = w[f"{phase}_flops"], w[f"{phase}_bytes"]
        print(f"{phase:<10}{f/1e9:>11.1f} GF{b/1024**2:>13.0f} MiB{f/b:>12.2f}")

    ratio = w["decode_bytes"] / w["prefill_bytes"]
    print(f"\nSame model. Decode moves {ratio:.0f}x more memory than prefill")
    print(f"and does only {w['decode_flops']/w['prefill_flops']:.1f}x the compute.")

    # Break the decode figure down, so it is never a bare number on screen.
    n_out = 256
    wb = w["weight_bytes"]
    weight_total = wb * n_out
    kv_total = w["decode_bytes"] - weight_total
    print("\nWhere decode's memory traffic comes from:")
    print(f"  weights, re-read once per token   "
          f"{wb/1024**2:6.0f} MiB x {n_out} = {weight_total/1024**2:10,.0f} MiB")
    print(f"  KV cache, re-read and growing     "
          f"{'':6}     {'':3}   {kv_total/1024**2:10,.0f} MiB")
    print(f"  {'':32}total   {w['decode_bytes']/1024**2:10,.0f} MiB")

    # Bandwidth floor for a single stream. Real engines land well under this.
    seconds = w["decode_bytes"] / RTX_3090_BANDWIDTH
    print(f"\n  ~= {w['decode_bytes']/1024**3:.0f} GiB to produce {n_out} tokens.")
    print(f"  At {RTX_3090_BANDWIDTH/1e9:.0f} GB/s that is {seconds:.2f}s minimum,"
          f" i.e. ~{n_out/seconds:.0f} tok/s")
    print("  for a single stream on a 3090. That is a CEILING, not a target --")
    print("  measure above it and your measurement is wrong.")

    print("\nWhy: prefill loads the weights ONCE for all 512 tokens.")
    print("Decode reloads all of them for EVERY ONE of the 256 tokens.")
    print("The shape to remember: decode traffic ~= weights x tokens generated.")

    print("\n\nCost PER TOKEN -- the honest comparison")
    print("=" * 62)
    print("Totals mislead: decode dominates every request simply because it")
    print("runs many steps. Divide by tokens produced and the gap is clearer.\n")
    print(f"{'prompt':>8}{'output':>8}{'prefill/tok':>14}{'decode/tok':>14}"
          f"{'ratio':>10}")
    for p, o in [(2048, 16), (512, 256), (128, 512), (32, 1024)]:
        w = phase_work(d, p, o)
        per_prefill = w["prefill_bytes"] / p       # bytes per prompt token
        per_decode = w["decode_bytes"] / o         # bytes per generated token
        print(f"{p:>8}{o:>8}{per_prefill/1024:>11.0f} KiB"
              f"{per_decode/1024**2:>11.0f} MiB{per_decode/per_prefill:>9.0f}x")

    print("\nA generated token costs orders of magnitude more memory traffic")
    print("than a prompt token -- and the ratio WIDENS as prompts get longer.")
    print("\nRead the two columns separately to see why:")
    print("  prefill/tok FALLS  -- one weight load amortized over more tokens")
    print("                        (the same effect as batching, along the prompt)")
    print("  decode/tok  is FLAT -- every step reloads every weight regardless")
    print("\nSo the gap is not fixed. It is set by how much you amortize.")

    print("\nIt also means request SHAPE decides which phase you should tune:")
    print("  summarization (long prompt, short answer) -> prefill-bound, TTFT")
    print("  chat and agents (short prompt, long answer) -> decode-bound, TPOT")
    print("Eventually they want different machines entirely (Lecture 27).")

    print("\n\nThe one that matters: batching")
    print("=" * 62)
    print("Decode reloads every weight per step. If 32 requests decode")
    print("TOGETHER, the weights are loaded once and serve all 32.\n")
    print(f"{'batch':>6}{'weight reads/token':>22}{'ops:byte':>12}"
          f"{'vs batch=1':>12}")
    params = model_params(d)
    wb = params * d.bytes_per_value
    base = None
    for bs in (1, 4, 16, 64, 256):
        flops = 2 * params * bs
        intensity = flops / wb          # weight traffic is FIXED
        base = base or intensity
        print(f"{bs:>6}{wb/1024**2:>17.0f} MiB{intensity:>12.1f}"
              f"{intensity/base:>11.0f}x")

    print("\nThe memory traffic never changes. The work does.")
    print("That is continuous batching (Lecture 08), and it is why serving")
    print("engines exist at all.")


if __name__ == "__main__":
    main()

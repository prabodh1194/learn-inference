"""Lecture 02 -- arithmetic intensity, by hand.

Runnable right now: pure arithmetic, no GPU and no model weights. Run it before
you implement anything else. Predicting a bottleneck on paper and then
confirming it by measurement is the habit the whole course is built on.

    uv run python book/code/roofline.py

Reproduces Kiely, *Inference Engineering* §2.4 (Figs 2.14-2.18, p.61-66) and
extends it to Qwen3-0.6B's real dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------


@dataclass
class Device:
    """A GPU's two numbers.

    IMPORTANT -- which FLOPS figure to use. Spec sheets list several and they
    differ by up to 8x:

      - "FP16 (half)" on TechPowerUp is the SHADER rate. The RTX 3090 shows
        35.58 TFLOPS "(1:1)", meaning fp16 runs at the same rate as fp32 on
        the shader cores.
      - The TENSOR CORE rate is what matmuls actually hit -- ~71 TFLOPS dense
        for the 3090 (fp16 with fp32 accumulate). This is the right number
        for inference, because every matmul in the model uses tensor cores.
      - Vendor marketing often quotes the SPARSE rate, which is 2x the dense
        figure and requires 2:4 structured sparsity you almost certainly
        don't have. Halve any suspiciously round number.

    Using the shader rate here would halve every ridge point and make decode
    look twice as close to compute-bound as it is.
    """

    name: str
    peak_flops: float      # FLOP/s, DENSE TENSOR CORE, at the dtype below
    bandwidth: float       # bytes/s
    dtype: str = "fp16"

    @property
    def ops_byte(self) -> float:
        """The ridge point. Below this intensity you are memory-bound."""
        return self.peak_flops / self.bandwidth


DEVICES = [
    # Book's reference example: 989 TFLOPS / 3.35 TB/s -> ~295 ops:byte
    Device("H100 SXM", 989e12, 3.35e12),
    Device("A100 80GB", 312e12, 2.039e12),
    Device("RTX 4090", 165e12, 1.008e12),
    # The rental target (Vast.ai, ~$0.20-0.25/hr). Ampere sm_86, 24GB.
    # 71 TFLOPS is the DENSE TENSOR CORE fp16 rate. Do not use TechPowerUp's
    # "FP16 (half) 35.58 TFLOPS (1:1)" -- that's the shader rate (see above).
    # Bandwidth 936.2 GB/s, verified against TechPowerUp.
    Device("RTX 3090", 71e12, 936.2e9),
    # M1 8-core GPU: ~2.6 TFLOPS fp16, ~68.25 GB/s unified memory.
    # Nominal figures -- measure your own in Lecture 02 and replace these.
    Device("Apple M1", 2.6e12, 68.25e9),
]


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


@dataclass
class ModelDims:
    """Qwen3-0.6B. Verified against the published config.json.

    https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
    """

    name: str = "Qwen3-0.6B"
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8        # GQA: fewer KV heads than Q heads -> smaller cache
    head_dim: int = 128        # note: 16*128 = 2048 != hidden. Qwen3 is like this.
    hidden: int = 1024
    intermediate: int = 3072   # SwiGLU MLP width (3x hidden, not the usual 4x)
    vocab: int = 151936
    bytes_per_value: int = 2   # fp16

    def kv_bytes_per_token(self) -> int:
        """K and V, per layer, across KV heads. GQA shrinks this vs. MHA."""
        return (
            2 * self.n_layers * self.n_kv_heads * self.head_dim * self.bytes_per_value
        )


def model_params(d: "ModelDims", include_embedding: bool = False) -> int:
    """Non-embedding parameter count, computed from the real architecture.

    Verified against Qwen3-0.6B's config.json: this gives 440.4M non-embedding
    (596.0M including the tied embedding, i.e. the "0.6B" in the name).

    The generic `12 * n_layers * hidden^2` shortcut you'll see in blog posts is
    20% LOW for this model, for two reasons worth knowing:

      - GQA makes QKV asymmetric. Q projects to n_heads*head_dim (2048) while
        K and V project to n_kv_heads*head_dim (1024) -- the shortcut assumes
        all three are hidden*hidden.
      - SwiGLU has THREE matrices (gate, up, down), and Qwen3's intermediate
        size is 3x hidden, not the 4x the shortcut assumes.

    Decode is memory-bound, so weight bytes set your decode time directly.
    A 20% error here is a 20% error in every decode prediction you make.
    """
    q_dim = d.n_heads * d.head_dim
    kv_dim = d.n_kv_heads * d.head_dim

    attn = (
        d.hidden * q_dim          # Q
        + 2 * d.hidden * kv_dim   # K, V (smaller under GQA)
        + q_dim * d.hidden        # O
    )
    mlp = 3 * d.hidden * d.intermediate   # gate, up, down

    total = d.n_layers * (attn + mlp)
    if include_embedding:
        total += d.vocab * d.hidden       # tied, so counted once
    return total


def model_params_bytes(d: "ModelDims") -> int:
    return model_params(d) * d.bytes_per_value


# --------------------------------------------------------------------------
# attention intensity -- book Figs 2.16-2.18
# --------------------------------------------------------------------------


def attention_intensity(n: int, d: int, bytes_per_value: int = 2) -> dict:
    """Unoptimized attention: S=QK^T, P=softmax(S), O=PV.

    Each step reads from HBM, computes, writes back -- including the full N x N
    score matrix, twice. That round-trip is exactly what FlashAttention removes
    (M2.3), which is why this number improves later.
    """
    b = bytes_per_value
    # reads + writes, per the book's table (Fig 2.15)
    memory = (
        (2 * b * n * d) + (b * n * n)          # S: read Q,K   write S
        + (b * n * n) + (b * n * n)            # P: read S     write P
        + (b * n * n) + (b * n * d) + (b * n * d)  # O: read P,V  write O
    )
    compute = (2 * d) * (n * n) + 3 * (n * n) + (2 * n) * (n * d)
    return {
        "n": n,
        "d": d,
        "memory_bytes": memory,
        "compute_ops": compute,
        "intensity": compute / memory,
    }


def decode_step_intensity(m: ModelDims, seq_len: int) -> float:
    """Whole-step intensity for generating ONE token.

    The dominant cost is re-loading every weight to produce a single token --
    a matrix-VECTOR product. Tiny compute, huge memory traffic. This is the
    number that makes decode memory-bound.
    """
    params = 12 * m.n_layers * m.hidden**2  # rough: attn + MLP projections
    weight_bytes = params * m.bytes_per_value
    kv_bytes = m.kv_bytes_per_token() * seq_len
    flops = 2 * params  # one matvec pass over the weights
    return flops / (weight_bytes + kv_bytes)


def prefill_intensity(m: ModelDims, n_tokens: int) -> float:
    """Same weights, but amortized over N tokens at once -- matrix-MATRIX.

    Weights are loaded once and reused across the whole prompt, so intensity
    scales with N. This is why prefill is compute-bound.
    """
    params = 12 * m.n_layers * m.hidden**2
    weight_bytes = params * m.bytes_per_value
    act_bytes = 2 * n_tokens * m.hidden * m.bytes_per_value
    flops = 2 * params * n_tokens
    return flops / (weight_bytes + act_bytes)


# --------------------------------------------------------------------------


def main() -> None:
    m = ModelDims()

    print(f"=== {m.name} ===")
    print(f"KV cache per token : {m.kv_bytes_per_token()} bytes "
          f"({m.kv_bytes_per_token() / 1024:.1f} KiB)")
    print(f"  -> 4096-token sequence: "
          f"{m.kv_bytes_per_token() * 4096 / 1024**2:.1f} MiB\n")

    print("=== hardware ridge points (ops:byte) ===")
    for dev in DEVICES:
        print(f"  {dev.name:<12} {dev.ops_byte:8.1f}    "
              f"({dev.peak_flops/1e12:.1f} TFLOPS / "
              f"{dev.bandwidth/1e9:.0f} GB/s)")

    print("\n=== book's worked example (N=4096, d=128) ===")
    r = attention_intensity(4096, 128)
    print(f"  memory    {r['memory_bytes']/1024**2:8.1f} MiB")
    print(f"  compute   {r['compute_ops']/1e9:8.1f} GFLOP")
    print(f"  intensity {r['intensity']:8.1f} ops:byte   (book says ~62)")

    print("\n=== attention intensity vs. sequence length (d=128) ===")
    for n in (128, 512, 1024, 2048, 4096, 8192):
        print(f"  N={n:<6} {attention_intensity(n, 128)['intensity']:7.1f} ops:byte")

    print("\n=== whole-step intensity, Qwen3-0.6B @ seq_len=2048 ===")
    dec = decode_step_intensity(m, 2048)
    print(f"  decode  (1 token)      {dec:8.2f} ops:byte")
    for n in (128, 512, 2048):
        print(f"  prefill ({n:>4} tokens)  "
              f"{prefill_intensity(m, n):8.2f} ops:byte")

    print("\n=== verdict ===")
    for dev in DEVICES:
        d_bound = "MEMORY" if dec < dev.ops_byte else "compute"
        p = prefill_intensity(m, 2048)
        p_bound = "memory" if p < dev.ops_byte else "COMPUTE"
        print(f"  {dev.name:<12} ridge={dev.ops_byte:6.1f}  "
              f"decode -> {d_bound:<6}  prefill(2048) -> {p_bound}")

    # Uncomment once you have measured achieved FLOP/s (needs M0.1 + M0.2):
    #
    # from bench.plot import roofline
    # dev = DEVICES[0]
    # roofline(
    #     peak_flops=dev.peak_flops,
    #     peak_bandwidth=dev.bandwidth,
    #     points={
    #         "decode (measured)":  (dec, MEASURED_DECODE_FLOPS),
    #         "prefill (measured)": (p,   MEASURED_PREFILL_FLOPS),
    #     },
    #     device_label=dev.name,
    # )


if __name__ == "__main__":
    main()

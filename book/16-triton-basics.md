# 16 — Triton basics

**Build:** `kernels/triton/softmax.py`, `kernels/triton/rmsnorm.py`
**Test:** `tests/test_16_triton.py` (cuda) · **Moves:** fewer memory round-trips on elementwise ops
**Prereq:** [15 — Profiling](15-profiling.md) — with your kernel ranking in hand

> **NVIDIA GPU required.** Triton is CUDA-only in practice.

---

## The problem

Your profile shows a long tail of small kernels — normalization, RoPE, activation,
residual adds. Individually trivial. Collectively 5–15% of decode.

They're slow for a reason you can now predict. Take RMSNorm in PyTorch:

```python
variance = x.pow(2).mean(-1, keepdim=True)   # read x, write variance
x = x * torch.rsqrt(variance + eps)          # read x, read variance, write x
return weight * x                            # read x, read weight, write out
```

Three operations, and `x` crosses memory **three times**. The arithmetic is
negligible; the traffic is everything. From Lecture 02, this is as memory-bound as
it gets.

Fuse them into one kernel and `x` is read once and written once.

---

## The idea

Triton is a Python DSL that compiles to GPU kernels. You write code describing
what **one block of threads** does; Triton handles thread indexing, memory
coalescing, and scheduling.

The mental model differs from CUDA in one important way:

> **CUDA:** you write what one *thread* does.
> **Triton:** you write what one *block* does, operating on tensors.

That's why a Triton kernel looks like NumPy with explicit loads and stores.

### Anatomy

```python
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols,
                   eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)              # which row am I?
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols             # guard the ragged edge

    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0)

    # everything below happens in registers/SRAM -- no HBM traffic
    variance = tl.sum(x * x, axis=0) / n_cols
    x_norm = x * tl.rsqrt(variance + eps)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

    tl.store(out_ptr + row * n_cols + offsets, x_norm * w, mask=mask)
```

Four things to internalize:

**`tl.program_id(0)`** — your block's index. Launch a grid of `n_rows` and each
instance handles one row.

**`BLOCK_SIZE: tl.constexpr`** — known at compile time, so Triton unrolls and
allocates registers accordingly. A different `BLOCK_SIZE` is a different compiled
kernel.

**`mask`** — vocabulary sizes aren't powers of two. `BLOCK_SIZE` is. The mask
suppresses out-of-bounds lanes, and `other=0.0` supplies a neutral value so
reductions stay correct.

**One load, one store.** Between them, everything is in registers. That's the
entire win.

---

## The exercises

### 1. Vector add

The "hello world". Get the launch grid and masking right; the kernel is trivial.
Won't beat PyTorch — it's already one fused kernel — but it teaches the mechanics.

### 2. Fused softmax

The first real one. Numerically stable softmax needs max, subtract, exp, sum,
divide — four passes over the row in PyTorch, one in Triton.

**Subtract the max before exponentiating.** `exp(x - max)` instead of `exp(x)`;
otherwise large logits overflow to `inf`. This is the same trick that becomes
*online softmax* in Lecture 17, so understand it here where it's simple.

### 3. RMSNorm

What Qwen3 actually uses. No mean subtraction, unlike LayerNorm — just
root-mean-square scaling. Straight from your profile's tail.

### Benchmark all three

Against the PyTorch built-in, at realistic sizes (`hidden=1024`, batch × seq
matching your workload). Report **achieved bandwidth as a fraction of peak** — the
Lecture 15 metric. A fused elementwise kernel should approach peak; if it doesn't,
your block size or launch grid is wrong.

### Autotuning

```python
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
             for bs in (64, 128, 256, 1024) for w in (2, 4, 8)],
    key=["n_cols"],
)
```

Triton benchmarks the configurations and caches the winner per `key`. Cheap to
add, and the optimum genuinely varies by GPU — which is a small lesson in itself
about portable performance.

---

## Build it

1. Vector add → softmax → RMSNorm, in that order.
2. `uv run pytest tests/test_16_triton.py -v` on a CUDA box.
   **`torch.allclose` against the PyTorch version is non-negotiable.** A kernel
   that's fast and wrong is worthless, and numerics bugs here are subtle — they
   show up as slightly worse output quality, not crashes.
3. Benchmark each against PyTorch. Record speedup **and** achieved bandwidth.
4. Swap RMSNorm into your engine. **Re-run the end-to-end benchmark.**

Step 4 is the point. Predict the end-to-end gain from your Lecture 15 profile
first: if RMSNorm was 4% of runtime and you make it 2× faster, you get 2%. Check
whether reality agrees — if it doesn't, your profile or your measurement is wrong,
and finding out which is worth more than the 2%.

---

## What you should see

**Softmax and RMSNorm meaningfully faster than PyTorch** — the fusion win is real.

**Vector add roughly tied.** Nothing to fuse.

**A small end-to-end gain**, matching what your profile predicted. Small is the
correct outcome; these kernels were a small share. Lectures 17–18 target the big
ones.

---

## Go deeper

- **[Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)** —
  02 (fused softmax) and 05 (layer norm) are directly this lecture. Work them.
- **[Triton: An Intermediate Language and Compiler for Tiled Neural Network
  Computations](https://dl.acm.org/doi/10.1145/3315508.3329973)** (Tillet et al.)
  — the design rationale for block-level programming.
- **Kiely §4.1.3** (p.100) — kernel fusion and reducing memory accesses.
- **vLLM `vllm/model_executor/layers/layernorm.py`** — production fused norms.

---

## Check yourself

1. Why is fused RMSNorm faster when the arithmetic is identical?
2. Why must `BLOCK_SIZE` be `tl.constexpr`?
3. What breaks without `mask` when `n_cols=1000` and `BLOCK_SIZE=1024`?
4. Why subtract the max before `exp`?
5. Your kernel is 3× faster but end-to-end improved 1.5%. Is that a
   disappointment or exactly what you predicted?

---

## Next

**[17 — FlashAttention](17-flash-attention.md)** — the kernel that actually
matters, and the deepest idea in Part III.

Two things to hold onto: it is **exact**, not an approximation; and it does
*more* arithmetic to move *less* data — which is the right trade on a
memory-bound operation, and the whole lesson of the roofline.

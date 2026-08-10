# 20. Raw CUDA

**Build:** `kernels/cuda/`, one kernel, by hand · **Test:** `tests/test_20_cuda.py` (cuda)
**Moves:** probably nothing. That's the lesson. · **Prereq:** [19. Quantization](19-quantization.md)

---

## The problem

Triton has been generating your kernels. It picks thread mappings, handles memory
coalescing, manages shared memory, and schedules loads.

That's a lot of decisions you haven't made, which means a lot you can't reason
about when a kernel underperforms. This lecture drops one level so those decisions
become visible.

**Set expectations honestly: you will probably not beat Triton.** It has years of
tuning behind its scheduling. The goal is understanding what it was doing on your
behalf, so its abstractions stop being magic and its failure modes become legible.

---

## The idea

CUDA's model is one level finer than Triton's:

| | Triton | CUDA |
|---|---|---|
| You write | what a **block** does | what a **thread** does |
| Indexing | `tl.program_id` | `blockIdx`, `threadIdx` |
| Memory | `tl.load` / `tl.store` | explicit `__shared__`, pointers |
| Coalescing | automatic | **your problem** |
| Sync | implicit | `__syncthreads()` |

### The four things that determine performance

**Coalescing.** Threads in a warp (32 threads) should read *consecutive* addresses,
so the hardware merges them into one transaction. Strided access can cost 32
transactions instead of 1, a 32× bandwidth penalty from an indexing choice that
looks innocent.

**Shared memory.** ~100× faster than HBM, ~100KB per SM, explicitly managed. This
is the resource FlashAttention's tiling exists to exploit; in Triton it was
implicit, here you allocate it yourself.

**Occupancy.** How many warps are resident per SM. Higher occupancy hides memory
latency by giving the scheduler other work. Registers and shared memory per block
bound it, use too much of either and occupancy collapses.

**Warp primitives.** `__shfl_down_sync` exchanges registers between threads in a
warp with no shared memory and no barrier. Reductions built this way are much
faster than the naive shared-memory version.

### A reduction, three ways

Worth writing all three: the progression *is* the lesson:

```cuda
// 1. naive: every thread hits shared memory, half idle immediately
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) sdata[tid] += sdata[tid + s];
    __syncthreads();
}

// 2. sequential addressing: no bank conflicts, no divergence
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}

// 3. warp shuffle: last 32 lanes need no shared memory or barriers
for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
```

Version 1 to version 3 is often several×. Every step is a memory-access or
divergence insight, not an arithmetic one, which by now should sound familiar.

---

## Build it

Pick **one** kernel. Softmax is the sane choice; paged attention is the ambitious
one.

1. Write it in `kernels/cuda/`, bind with `torch.utils.cpp_extension.load`.
2. `uv run pytest tests/test_20_cuda.py -v`, correctness first, as always.
3. Profile with `ncu`. Record: achieved bandwidth vs. peak, occupancy, warp
   efficiency.
4. **Compare against your Triton version**, and against PyTorch.
5. If Triton wins, use `ncu` to find out *why*, usually better memory pipelining
   or a smarter thread mapping than you chose.

Then do the reduction progression above and measure each stage. It's the cheapest
way to internalize why memory access patterns dominate.

---

## What you should see

**Triton probably wins**, and the gap tells you what it was doing for you.

**Your first version much slower**, most likely from uncoalesced access. Finding
that in `ncu` is the single most valuable exercise here.

**Big gains from small changes**: the reduction progression makes this vivid.

**A new ability:** you can now read CUDA in vLLM (`csrc/`) and follow it. That's
the durable outcome.

---

## When raw CUDA is actually worth it

Not often, and it's worth being clear about when:

- **Novel algorithms** Triton can't express well (unusual memory patterns,
  specialized warp cooperation).
- **The last 10–20%** on a kernel that dominates your profile.
- **Hardware-specific features**: tensor core instructions, async copy, TMA on
  Hopper.

For everything else, Triton is a better default: dramatically less code, portable
across architectures, and usually within a small factor. FlashAttention itself is
hand-written CUDA *and* has a Triton implementation: the fact that both exist is
the honest summary of this tradeoff.

---

## Go deeper

- **[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)**:
  chapters 3 (model) and 5 (performance).
- **[Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)**
  (Mark Harris): the classic seven-stage walkthrough. Still the best single
  document on why memory access dominates.
- **vLLM `csrc/`**: the remaining hand-written CUDA. Note that V1 moved paged
  attention itself to Triton, so `csrc/attention/` now holds mostly headers; the
  old `paged_attention_v1.cu` lives only in pre-V1 tags. A good reminder that
  hand-written CUDA gets replaced when a portable version gets close enough.
- **Kiely §4.1–4.1.3** (p.96–100), CUDA kernels, selection, and fusion.

---

## Check yourself

1. Why can uncoalesced access cost 32× bandwidth?
2. What limits occupancy, and why doesn't maximum occupancy always mean maximum
   speed?
3. Why is `__shfl_down_sync` faster than a shared-memory reduction for the last 32
   elements?
4. Triton beat your CUDA. From `ncu`, what specifically did it do better?
5. Name a case where hand-written CUDA is genuinely worth it, and one where
   reaching for it is a mistake.

---

**Part III complete.** You've profiled, written Triton, implemented
FlashAttention and paged attention, quantized with a real quality bar, and gone
down to raw CUDA.

## Next

**[21. JAX and XLA](21-jax-and-xla.md)**: a different way to think about all
of this: you declare *what*, the compiler decides *how*.

Verify your JAX forward pass against PyTorch **before** moving on. A silent
transcription bug in RoPE or attention will cost you hours in L22, where you'd
also be debugging sharding.

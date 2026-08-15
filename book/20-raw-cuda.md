# 20. Raw CUDA

**Build:** `kernels/cuda/`, one kernel, by hand · **Test:** `tests/test_20_cuda.py` (cuda)
**Moves:** probably nothing. That's the lesson. · **Prereq:** [19. Quantization](19-quantization.md)

---

## The problem

Recall what Triton has been doing for you (Lecture 16): you described what one
**block** does, a block being a group of threads that run together on one SM,
one of the chip's work groups, each with its own fast private memory. Triton
filled in the rest. It picks thread mappings (which thread handles which piece
of data), handles memory coalescing (arranging those reads so the hardware
does them efficiently), manages shared memory (the SM's private fast memory,
which you never touched in Triton), and schedules loads (deciding when bytes
are fetched so no one waits).

That's a lot of decisions you haven't made, which means a lot you can't reason
about when a kernel underperforms. When a Triton kernel is slow, you cannot
point at a decision and say: this one is wrong. The options are invisible. This
lecture drops one level so those decisions become visible. You write them
yourself, even badly, and the badness teaches you what they were for.

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

Before the list of four, the situation to keep in your head: a GPU
runs many threads at once, and they all need data. The memory that holds the
data (HBM, the GPU's main RAM) is far away and slow compared to the chip's own
work area. Almost every performance question reduces to one of two things: how
many separate trips to that slow memory your access pattern forces, and how
much of the chip stays busy while trips are in flight. The four entries below
are those two questions in their technical dress.

**Coalescing.** Threads in a warp (32 threads that execute the same instruction
together, in lockstep) should read *consecutive* addresses,
so the hardware merges them into one transaction, a single exchange with the
memory chips. Strided access can cost 32
transactions instead of 1, a 32× bandwidth penalty from an indexing choice that
looks innocent. Draw it:

```
coalesced:  thread  0   1    2   ...  31
            reads   0   1    2   ...  31      one transaction: the chips hand
            (a row of an array)                over one long run of bytes

strided:    thread  0   1    2   ...  31
            reads   0  1024 2048 ... 31744    32 transactions: each thread
            (a column, or every 1024th)        needs its own separate visit
```

Both loops are just `data[tid * stride]` versus `data[tid]`. The innocent
expression is the strided one. The tell: your indexing expression jumps, the
hardware pays per jump.

**Shared memory.** The fast private memory attached to each SM, ~20× faster than
HBM on the 3090, ~100 KB per SM, explicitly managed. This
is the resource FlashAttention's tiling exists to exploit; in Triton it was
implicit, here you allocate it yourself. It exists to let the threads of one
block hand data to each other and to hold a working set close, both without
going to the slow main memory.

**Occupancy.** How many warps are resident per SM, parked and ready. Higher
occupancy hides memory
latency by giving the scheduler other work. The mechanism is ordinary
concurrency: a warp that asks for data must wait for it, and during the wait
its SM is capable of doing nothing. If other warps are ready, the SM runs them
instead, and the wait costs nothing. Few resident warps, nothing to run while
the first waits, the pipeline drains. Registers and shared memory per block
bound it, use too much of either and occupancy collapses.

**Warp primitives.** `__shfl_down_sync` lets one thread pick a value straight
out of another thread's private registers, no shared memory and no barrier (a
barrier being a point where every thread must arrive before any thread may
leave). A **reduction** is combining many numbers into one, a sum over a
sequence, the classic parallel exercise. Reductions built with shuffles are
much faster than the naive shared-memory version, because the data never
touches memory at all.

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

Two words in those comments are jargon worth unpacking. **Bank conflicts**:
shared memory is built from 32 parallel banks, one per lane; when two threads
in a warp touch the same bank at once, the hardware serves them one at a time,
quietly turning one access into two or more. **Divergence**: threads in a warp
execute the same instruction together, so when some threads take a branch and
others don't, both sides run, with half the threads idle. `if (tid % (2*s) == 0)`
and `if (tid < s)` are the same arithmetic; only the first one divides the warp
into interleaved camps, and adds veiled idleness on top of the bank conflicts.

Read the three versions as a sequence of errors being removed. Version 1 makes
every-thread-but-one share the same red-hot banks and has the warp branch apart
each step. Version 2 fixes both by making the threads that act a contiguous
leading slice, and the active threads all touch distinct banks. Version 3
notices that the last 32 values all live in registers of threads that are
already executing together, so they can be merged with no memory at all.

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

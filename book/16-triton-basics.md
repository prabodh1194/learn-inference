# 16. Triton basics

**Build:** `kernels/triton/softmax.py`, `kernels/triton/rmsnorm.py` · **Test:** `tests/test_16_triton.py` (cuda) · **Moves:** fewer memory round-trips on elementwise ops
**Prereq:** [15. Profiling](15-profiling.md), with your kernel ranking in hand

> **NVIDIA GPU required.** Triton is CUDA-only in practice.

---

## The problem

Your profile shows a long tail of small kernels, normalization, RoPE, activation,
residual adds. Individually trivial. Collectively 5–15% of decode (Lecture 15's
profile table, the "Norms, RoPE, elementwise" row).

They're slow for a reason you can now predict. Take RMSNorm in PyTorch:

```python
variance = x.pow(2).mean(-1, keepdim=True)   # read x, write variance
x = x * torch.rsqrt(variance + eps)          # read x, read variance, write x
return weight * x                            # read x, read weight, write out
```

Three operations, and `x` crosses memory **three times**. The arithmetic is
negligible; the traffic is everything. Count the bytes for one row of 1024
values, our hidden size, in fp16 (2 bytes per value, so 2 KiB per row):

```
pass 1:  read x                                       2 KiB
pass 2:  read x,  write x                             2 + 2   =  4 KiB
pass 3:  read x,  read weight,  write out             2 + 2 + 2  =  6 KiB
                                                        total     = 12 KiB
```

(The variance is one value per row, 4 bytes, written then re-read: negligible
next to the row itself, so it is not counted here.)

Twelve kilobytes of traffic per row, to compute a mean and a scale. From
Lecture 02, this is as memory-bound as it gets. The one fused kernel reads `x`
once, reads `weight` once, writes the output once:

```
fused:   read x,  read weight,  write out             2 + 2 + 2  =  6 KiB
```

Half the traffic, for the same arithmetic. Fuse them into one kernel and `x` is
read once and written once.

??? question "Why doesn't eager PyTorch fuse these three ops itself?"
    Because eager PyTorch runs each operation as its own kernel, written out to
    memory before the next one starts, and nothing tells it the intermediate
    can stay on-chip instead. Fusion is a decision a compiler must make, and
    Triton lets you make it explicitly, by writing one kernel instead of three.
    (Lecture 13's `torch.compile` automates exactly this decision, which is why
    the two techniques compound.)
    [Full answer](qa.md#why-doesnt-eager-pytorch-fuse-these-three-ops-itself)

---

## The mental model

Before the code, the map. There are three ways to run math on a GPU, and
knowing which one you're in is half the battle:

| | Eager PyTorch | Raw CUDA | Triton |
|---|---|---|---|
| What you write | Python ops | what **one thread** does | what **one block** does, on tensors |
| Kernel launches | one per op | one per kernel you write | one per kernel you write |
| Compiler/bookkeeping | — | you manage it all | indexing, coalescing, scheduling |
| Control over hardware | none | total | block-level; the rest hidden |

**Eager PyTorch** launches a kernel per operation, and the launch overhead and
the memory round-trip between them is Lecture 13's story and the 12 KiB above.
**Raw CUDA** (Lecture 20) gives you everything, at the price of writing a
thread's worth of work and managing the machine yourself. **Triton** sits
between: you write a kernel like a NumPy function, and the compiler does the
bookkeeping a CUDA programmer does by hand.

That bookkeeping is concrete, and naming it demystifies the abstraction:

- **Thread indexing** — which thread reads which data element.
- **Memory coalescing** — arranging neighbouring threads to read neighbouring
  addresses, so the hardware fetches them in one transaction instead of many.
- **Scheduling** — which warps (the chip's fixed 32-thread groups) run when,
  and how the grid of blocks drains onto the machine.

The mental model differs from CUDA in one important way:

> **CUDA:** you write what one *thread* does.
> **Triton:** you write what one *block* does, operating on tensors.

That's why a Triton kernel looks like NumPy with explicit loads and stores.

Two consequences that shape everything that follows:

**It's a compiler, and it compiles the whole body.** The kernel you write is
compiled to one launch — one CPU-to-GPU dispatch, not one per `tl.*` line. All
the lines between your first `tl.load` and your final `tl.store` become
register-level arithmetic that never touches memory. That fusion, in one
launch, is the entire win this lecture is about.

**It's a block language, not a thread language.** The `tl.*` functions operate
on the block's tensor — `tl.sum(x * x, axis=0)` reduces the block, `x * w`
broadcasts a scalar across it. The compiler turns those tensor operations into
per-thread instructions and reductions across threads (shared memory and warp
shuffles). You express the shape of the work; it does the threading.

### When Triton wins — and when it can't

Honest boundaries, so you don't reach for it in the wrong place.

**It wins** on memory-bound, shape-regular work: elementwise and reduction
patterns (norms, activations, the fused attention tiles of Lecture 17). The
win is removing HBM round-trips, and Triton's one-launch fused body is exactly
that. It wins for anything you'd happily write as a small CUDA kernel and
settle for 80% of peak.

**It can't fix the algorithm.** FlashAttention's speedup is an *idea* (never
materialize the score matrix) — Triton is only the tool that implements it.
If a kernel is slow because of an algorithmic choice, no kernel language fixes
that; you have to change the algorithm first (Lecture 17).

**It won't beat a kernel already at the roofline.** Lecture 15's ceiling check
applies. If your kernel is at 85% of peak bandwidth, Triton has nothing to add;
the win was taken before you started.

**Dynamic shapes are awkward.** `BLOCK_SIZE` is a power-of-two compile-time
constant. A different shape is a different compiled kernel, and a shape that
changes per call recompiles (or autotunes, below). Ragged tails are handled
with masks, not dynamic loop bounds.

**The last 5% is hidden.** Register allocation, warp specialization (different
warps doing different jobs), explicit pipelining (overlapping the next tile's
fetch with the current tile's math) are under the compiler's control. When an
official kernel beats yours by 2×, the gap is usually exactly these — which is
why Lecture 17 warns you not to expect to beat it, and why Lecture 20's raw
CUDA exists.

**You still need the profiler's view.** Triton hides the hardware, so occupancy
and achieved bandwidth (Lecture 15) are still your windows into what's really
happening. The abstraction makes writing kernels cheap; it doesn't make
reading the machine unnecessary.

---

## Close reading: three kernels

The three kernels below are the lecture. Read each one twice: once for the
pattern it teaches, once for the annotations. Then type them into the stubs in
`kernels/triton/` and the build section has what to do next.

### The skeleton every kernel shares

Four pieces, in this order, every time:

1. **Launch config** — the `@triton.jit` decorator and the grid (how many
   blocks, and with what compile-time constants).
2. **Compute your addresses** — `tl.program_id` + `tl.arange` + offsets.
3. **Load** — `tl.load` brings data into registers.
4. **Compute, then store** — everything between the load and the store lives
   on-chip; the store is the only write.

### 1. Vector add

The "hello world" — no reduction, no mask subtleties, just the skeleton:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                                  # (1) which block am I?
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # (2) my slice of the array
    mask = offsets < n_elements                             #     guard the tail

    x = tl.load(x_ptr + offsets, mask=mask)                 # (3) one coalesced read
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)           # (4) one coalesced write

grid = (triton.cdiv(n_elements, BLOCK_SIZE),)               # enough blocks to cover the array
add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
```

- **(1)** `tl.program_id(0)` is the block's index, in `[0, grid)`. The grid
  declares how many blocks exist; each block computes its own slice from its id.

??? question "Is `pid` the same as a process ID?"
    No — it's **program id**, and it's nothing to do with the operating
    system. Triton compiles your kernel to one device program, then
    instantiates many copies of it, one per block (the paper calls them
    "program instances"). `pid` is the index of *this copy* among all of them
    in the launch — the same role as CUDA's `blockIdx.x`. An OS process id
    identifies a running process on your machine's scheduler; this identifies
    a work unit inside one GPU call. It's a number chosen before the kernel
    starts, and its only job is addressing.
    [Full answer](qa.md#is-pid-the-same-as-a-process-id)
- **(2)** `tl.arange(0, BLOCK_SIZE)` is a **vector of offsets**, `[0, 1, …,
  BLOCK_SIZE-1]`, held in registers. Add the block's starting offset and you
  have the addresses this block owns. This is where the compiler's coalescing
  happens: the block's lanes map to consecutive addresses, so the hardware
  fetches the whole slice in a handful of transactions.
- **(3)** `tl.load` reads a whole tensor of values into registers. The mask
  suppresses out-of-bounds lanes; masked-out lanes read undefined values, which
  is fine here because the store is masked too — undefined lanes are never
  written.
- **(4)** `x + y` is one elementwise add across the block, all in registers.
  One `tl.store`, masked, and the kernel is done.

`triton.cdiv(n, block)` is "divide, rounding up": `n=1000, block=1024` → `1`,
`n=3000, block=1024` → `3`. Exactly enough blocks to cover the array, no
overshoot.

#### See it: grid → block → offsets → state

The four concepts land together in a picture. Take `n = 3000`,
`BLOCK_SIZE = 1024`, so `grid = (3,)` — three blocks.

**The grid splits the array.** Each block owns one contiguous span:

```
block 0        ┌───────────────────────────────────┐  pid = 0
               │ indices       0 .. 1023           │  1024 real elements
               └───────────────────────────────────┘
block 1        ┌───────────────────────────────────┐  pid = 1
               │ indices    1024 .. 2047           │  1024 real elements
               └───────────────────────────────────┘
block 2        ┌───────────────────────┬───────────┐  pid = 2
               │ indices   2048 .. 2999│ 3000..3071│  952 real + 72 masked
               └───────────────────────┴───────────┘
```

**A block turns `pid` into addresses.** `offsets = pid·BLOCK_SIZE + tl.arange(0,
BLOCK_SIZE)` — the block's starting point, plus a per-lane counter. Block 1:

```
                  lane      0      1      2    ...    1023
  tl.arange        →        0      1      2    ...    1023
  pid · BLOCK      →     1024   1024   1024    ...    1024
                            ↓      ↓      ↓             ↓
  offsets          =    1024   1025   1026    ...     2047
                      x[1024] x[1025] x[1026]        x[2047]
```

`tl.arange` is a register vector, not a memory access — it's the compiler's
way of giving each lane its identity, then the address is `x_ptr + offset`.

**The state transition, lane by lane.** Every lane runs the same four states,
in lockstep. A few lanes shown, all 1024 do it together:

```
   state       lane 0       lane 1      lane 2      ...    lane 1023
   ───────────────────────────────────────────────────────────────────
   offsets      1024        1025        1026        ...       2047
   load x      x[1024]     x[1025]     x[1026]      ...     x[2047]
   load y      y[1024]     y[1025]     y[1026]      ...     y[2047]
   x + y        each lane adds its two loaded values, in registers
   store      out[1024]   out[1025]   out[1026]     ...    out[2047]
```

Read the table top to bottom and it's one lane's timeline. Of the five
states, **three touch memory**: the two loads at the top (the block's inputs
arriving from HBM) and the store at the bottom (the output leaving for HBM).
The other two — computing `offsets`, and the add itself — are pure register
arithmetic: they move data that is already on the chip, and not one byte
goes to or from memory.

That's the boundary view of the whole kernel. The memory system sees exactly
three spans from this block — `x[1024..2047]`, `y[1024..2047]`,
`out[1024..2047]` — a 2 KiB read, a 2 KiB read, a 2 KiB write, 6 KiB total.
Everything else in the table is invisible to it. Which is the whole point of
the fused kernel: PyTorch's three separate operations made `x` cross memory
three times (the 12 KiB at the top of the lecture); this kernel crosses
once. The middle rows not touching memory is not a detail — it is the
entire win.

**The masked tail.** Block 2's span is wider than the array, so its last 72
lanes are fenced off. The mask is a per-lane bit; `other` is what the fenced
lanes read so they don't poison the math:

```
  offsets:    2048  2049  ...  2999 │ 3000  3001  ...  3071
              mask=1  mask=1 ... mask=1 │ mask=0 mask=0 ... mask=0
              ─────── 952 real lanes ──┼──── 72 masked lanes ────
  load x:     read, kept               │ read, discarded
  store:      out[2048..2999] written  │ lane skipped, nothing written
```

**Why the loads are cheap.** The block's 1024 lanes hit 1024 *consecutive*
addresses — one contiguous span in memory — so the hardware serves them as a
handful of wide transactions instead of 1024 separate round-trips. That is
the coalescing the compiler arranges for you, and it's the difference between
a memory-bound kernel at peak bandwidth and one at 20% (Lecture 15).

### 2. Fused softmax

The first real one, and the point of the lecture: numerically stable softmax
is four passes over the row in PyTorch (max, subtract, exp, sum, divide), one
in Triton. A pass is a full read of the row from memory, so the row crosses
memory four times for an operation that is one read, one scaling, one write.

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols                      # ragged row: 1000 real, 24 pad

    x = tl.load(x_ptr + row * n_cols + offsets,  # the row's BLOCK_SIZE values
                mask=mask, other=-float("inf"))

    m = tl.max(x, axis=0)                        # the row's max — a scalar
    p = tl.exp(x - m)                            # subtract-max, then exp: broadcasts
    l = tl.sum(p, axis=0)                        # the row's sum — another scalar
    out = p / l

    tl.store(out_ptr + row * n_cols + offsets, out, mask=mask)
```

Four things here that matter more than the code:

**Why `other=-float("inf")`, not `0.0`.** The masked lanes (24 of them, the
pad) participate in the `tl.max` and `tl.sum`. If `other=0.0`, the max is
unaffected, but the sum gets `24 × exp(0 − m) = 24 × e^−m` of junk — the
softmax denominator is slightly too big, the row is slightly off. With
`other=-inf`, the masked lanes contribute `exp(-inf − m) = 0` to the sum and
never win the max: the math is exactly right, pad lanes add nothing.

**Subtract the max before exponentiating.** `exp(x − max)` instead of `exp(x)`;
otherwise large logits overflow to `inf`. This is the same trick that becomes
*online softmax* in Lecture 17, so understand it here where it's simple.

**The reduction collapses the block to a scalar.** `tl.max(x, axis=0)` reduces
a `[BLOCK_SIZE]` tensor to one value. The compiler implements it as reductions
across the block's warps (shuffles plus shared memory). The scalar then
**broadcasts** back: `x - m` and `p / l` apply a single value to every lane.
One reduction down, one broadcast up — that shape is the fingerprint of a
row-wise kernel, and you'll see it in RMSNorm right now.

The same state-transition view, compressed — a `[BLOCK]` row collapses to a
scalar, then spreads back:

```
  lane           0       1       2    ...   1023
  x        x[1024]  x[1025]  x[1026]  ...  x[2047]
                    └── tl.max(x, axis=0) ──┘
                          ▼
  m              (one scalar: the row's max)
                    ┌──── broadcasts back ────┐
  p        e⁰     e⁻¹     e⁻²   ...    e⁻³        exp(x − m), per lane
                    └── tl.sum(p, axis=0) ───┘
                          ▼
  l              (one scalar: the row's sum)
                    ┌──── broadcasts back ────┐
  out      p0/l    p1/l    p2/l   ...   p1023/l
```

The two reductions never materialize anything bigger than the row — the
collapsed scalars live in registers and spread right back out. No HBM traffic
between the row and its scalars.

**Why this beats PyTorch.** Four passes over the row in eager mode, one pass
here. The four rounds of round-tripping become zero — the max, subtract, exp,
sum, and divide all happen in registers on the one load. Same math, same
result, a quarter of the traffic.

### 3. RMSNorm

What Qwen3 actually uses. No mean subtraction, unlike LayerNorm, just
root-mean-square scaling. Straight from your profile's tail. Two loads, one
store — the entire 12 KiB problem at the top of the lecture, reduced to 6 KiB:

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols,
                   eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)              # which row am I?
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols             # guard the ragged edge

    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0)

    variance = tl.sum(x * x, axis=0) / n_cols   # reduce, in registers
    x_norm = x * tl.rsqrt(variance + eps)       # scalar broadcast: x * (1/√var)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

    tl.store(out_ptr + row * n_cols + offsets, x_norm * w, mask=mask)
```

**Note the `other=0.0` here — deliberately different from softmax's `-inf`.**
This reduction is a sum of squares; a masked lane *should* contribute 0, that's
exactly what "not part of the row" means. The neutral value isn't a magic
number — it's whatever makes the reduction correct for masked-out lanes, and
it depends on the reduction. Sum of squares: `0`. `exp` of a max-subtracted
value: `-inf`. That one line is the whole lesson about masking.

**Two loads, one store.** `x` and `w` are each read once, the output written
once. `variance`, `x_norm`, and the product all live in registers between
them — the sum of squares is reduced on-chip, the scaling is applied on-chip.
The arithmetic is identical to the three-PyTorch-op version; the traffic is
half. This is the fused kernel that "just" fused.

### Autotuning

```python
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
             for bs in (64, 128, 256, 1024) for w in (2, 4, 8)],
    key=["n_cols"],
)
```

Triton benchmarks the configurations and caches the winner per `key`. Cheap to
add, and the optimum genuinely varies by GPU, which is a small lesson in itself
about portable performance. (`num_warps` is how many warps, the chip's fixed
32-thread groups, the block is split into.) Because each configuration is a
*different compiled kernel* — different unrolling, different register
allocation — autotuning is really "try the whole design space of compiled
versions, keep the fastest." It is the direct answer to the dynamic-shape
problem from the mental model section: shapes don't have to be one fixed
constant, they just have to be *chosen from a fixed set the compiler has
already prepared*.

---

## Build it

1. Type the three annotated kernels above into `kernels/triton/` — `softmax.py`
   and `rmsnorm.py` are stubs waiting for exactly this code.
2. Vector add → softmax → RMSNorm, in that order. (Vector add won't beat
   PyTorch — it's already one fused kernel — but it teaches the launch and the
   mask before the real ones.)
3. `uv run pytest tests/test_16_triton.py -v` on a CUDA box.
   **`torch.allclose` against the PyTorch version is non-negotiable.** A kernel
   that's fast and wrong is worthless, and numerics bugs here are subtle, they
   show up as slightly worse output quality, not crashes.
4. Benchmark each against PyTorch. Record speedup **and** achieved bandwidth
   as a fraction of peak, the Lecture 15 metric. A fused elementwise kernel
   should approach peak; if it doesn't, your block size or launch grid is wrong.
5. Swap RMSNorm into your engine. **Re-run the end-to-end benchmark.**

Step 5 is the point. Predict the end-to-end gain from your Lecture 15 profile
first: if RMSNorm was 4% of runtime and you make it 2× faster, you get 2%. Check
whether reality agrees, if it doesn't, your profile or your measurement is wrong,
and finding out which is worth more than the 2%.

---

## What you should see

**Softmax and RMSNorm meaningfully faster than PyTorch**: the fusion win is real.

**Vector add roughly tied.** Nothing to fuse.

**A small end-to-end gain**, matching what your profile predicted. Small is the
correct outcome; these kernels were a small share. Lectures 17–18 target the big
ones.

---

## Go deeper

- **[Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)**:   02 (fused softmax) and 05 (layer norm) are directly this lecture. Work them.
- **[Triton: An Intermediate Language and Compiler for Tiled Neural Network
  Computations](https://dl.acm.org/doi/10.1145/3315508.3329973)** (Tillet et al.)
  : the design rationale for block-level programming.
- **Kiely §4.1.3** (p.100), kernel fusion and reducing memory accesses.
- **vLLM `vllm/model_executor/layers/layernorm.py`**: production fused norms.

---

## Check yourself

1. Why is fused RMSNorm faster when the arithmetic is identical?
2. Why must `BLOCK_SIZE` be `tl.constexpr`? (Hint: what changes when it is?)
3. What breaks without `mask` when `n_cols=1000` and `BLOCK_SIZE=1024`?
4. Softmax masks with `other=-inf`, RMSNorm with `other=0.0`. Why are they
   different, and what rule decides which you need?
5. Why subtract the max before `exp`?
6. Your kernel is 3× faster but end-to-end improved 1.5%. Is that a
   disappointment or exactly what you predicted?
7. A Triton kernel is at 20% of peak bandwidth and correct. Is that a Triton
   failure? (Lecture 15.)

---

## Next

**[17. FlashAttention](17-flash-attention.md)**: the kernel that actually
matters, and the deepest idea in Part III.

Two things to hold onto: it is **exact**, not an approximation; and it does
*more* arithmetic to move *less* data, which is the right trade on a
memory-bound operation, and the whole lesson of the roofline.

The mental model from this lecture carries straight over: FlashAttention is the
algorithm (never materialize the score matrix), and the Triton you just learned
is the tool that implements it — blocks, tiles, reductions, and the same
one-load-one-store discipline, on a 2-D tile instead of a row.

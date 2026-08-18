# 22. Tensor parallelism

**Build:** `jaxlm/sharding.py`, then TP by hand in `engine/`
**Test:** `tests/test_22_tp.py` (cuda, multi-GPU) · **Moves:** per-user latency, and reveals where scaling stops
**Prereq:** [21. JAX and XLA](21-jax-and-xla.md)

> **Two or more GPUs required.** On Vast.ai, a 2×3090 box. Sharding *logic* can be
> reasoned about on one device; scaling curves cannot.

---

## The problem

First, what "more than one GPU" means here, because packaging has made the
word slippery. **A GPU is a separate memory space.** Two GPUs cannot read each
other's tensors; combining anything requires a collective over a link. That is
the property this whole lecture is about — not how many chips are in the box.

```
   ONE GPU                        MULTI-GPU
   1 CUDA device                  2+ CUDA devices
   one address space              separate address spaces
   a[i] just works                needs all-reduce / all-gather
   no NCCL, no TP                 this lecture
```

A single **B200** is one GPU by that test: your code sees one device and 192 GB
of unified memory, and you never write a collective — even though the package
physically holds two dies joined by a ~10 TB/s interconnect. Multi-*die*,
single-GPU. Two B200s in a box is multi-GPU; so is one GPU per node across a
cluster, only slower.

With that fixed, two reasons to use more than one:

**The model doesn't fit.** 70B in FP16 is 140 GB: `70 × 10⁹ params × 2 bytes
(fp16) = 140 GB`. Whether that needs multiple GPUs depends entirely on the card
in front of you:

```
   RTX 3090     24 GB   →  needs 6+ GPUs
   A100         80 GB   →  needs 2
   H100         80 GB   →  needs 2
   B200        192 GB   →  fits on ONE
```

This reason is receding as memory grows, and on current hardware plenty of
models that once demanded TP now fit on one card. The second reason does not
recede.

**One GPU is too slow for one user.** Decode is memory-bound (Lecture 02), so a
single user's tokens/sec is capped by *one* GPU's bandwidth: every token moves
the active weights, and the token can leave no faster than the memory can hand
those bytes over. Batching raises aggregate
throughput (tokens per second across all users) but never helps that one user. Splitting the weights across
GPUs multiplies the bandwidth available to a single sequence: each GPU now
carries a slice of the weights, each slice loaded in parallel, so the bytes the
sequence must wait for are divided by the number of GPUs. Same memory, more
taps.

That second reason is why TP is a **latency** optimization, and it's the one people
miss.

---

## The idea

You have a model too big for one GPU, or one that fits but decodes too slowly.
Either way you now have several GPUs and one question: **what do you cut, and
where?**

A transformer gives you three natural seams, and they are genuinely different
decisions rather than three flavours of the same one:

```
   the model                  cut it...

   ┌───────────┐   layer 1     ─── horizontally ───►  PIPELINE (PP)
   ├───────────┤   layer 2                            GPU 0 gets layers 1-14
   ├───────────┤   ...                                GPU 1 gets layers 15-28
   ├───────────┤   layer 28
   └───────────┘

   ┌─────┬─────┐               ─── vertically ─────►  TENSOR (TP)
   │  W  │  W  │   every layer                        every GPU holds a SLICE
   │ half│ half│                                     of every layer
   └─────┴─────┘

   ┌───┬───┬───┐               ─── by expert ──────►  EXPERT (EP)
   │ e │ e │ e │   MoE only                           each GPU owns whole
   └───┴───┴───┘                                      experts (Lecture 23)
```

**This lecture is about the vertical cut.** Here is why, in one line each:

- **Pipeline** splits by layer. Cheap to communicate — one activation handoff
  per stage boundary — but each GPU only works on part of the model, so a token
  still walks through all of them in sequence. It does not make a single token
  faster.
- **Tensor** splits *inside* every layer, so all GPUs work on every token at
  once. That is what actually reduces per-token latency — and it is why this is
  the default within a machine.
- **Expert** applies only to mixture-of-experts models, and is Lecture 23.

The price of the vertical cut is that a slice of a layer produces a *partial*
answer. The GPUs must combine their partials before the next layer can start,
which they do with an **all-reduce**: every GPU contributes its piece, every GPU
receives the total. That collective, once or twice per layer, is essentially the
entire cost of TP — and most of the rest of this lecture is about how expensive
it gets.

??? note "Why not pipeline parallelism for inference?"
    The usual objection to PP is *bubbles* — idle stretches where a stage waits
    for work from the stage behind it. That is a **training** problem: there,
    micro-batches queue behind each other and stages sit idle. In inference the
    stages fill back-to-back, so PP is roughly latency-neutral.

    The real reasons inference avoids it are different. PP **cannot shard the KV
    cache**: each stage needs the cache for its own layers, and the number of
    in-flight sequences rises with the number of stages, which cancels the
    weight-memory win you bought. And each stage boundary adds a hop of latency
    that compounds across machines. In practice: at most 1–2 stages, if any.

### How the split works

Start from a fact about matmuls that makes the whole scheme possible.

```
   out[i,j] = Σ over k   x[i,k] · W[k,j]
                ▲          ▲        ▲
                │          │        └── column j of W
                │          └── row i of x
                └── the shared inner dimension
```

**Every output element depends only on one row of `x` and one column of `W`.**
No output element needs any other output element. So in principle all of them
could be computed at once — a matmul is embarrassingly parallel by construction,
and always has been. A single GPU already exploits this across its thousands of
cores.

That means **TP is not finding parallelism. The parallelism was free.** What TP
chooses is *which axis to spread across separate memory spaces* — and that
choice is the entire subject, because the axes are not equivalent:

```
   split along j (columns of W)          split along k (the shared sum)

   GPU 0 ← W[:, 0:1536]                  GPU 0 ← W[0:1536, :]
   GPU 1 ← W[:, 1536:3072]               GPU 1 ← W[1536:3072, :]

   each GPU computes COMPLETE            each GPU computes a PARTIAL sum:
   outputs for its own columns           its terms of Σ_k, missing the rest

   ✓ nothing to combine                  ✗ must all-reduce before use
```

Splitting along `j` is free because the columns never interact. Splitting along
`k` cuts *through* the summation, so each GPU holds an incomplete answer.

The elegance of the MLP is that its two matmuls can be split on complementary
axes, so you pay the `k`-axis cost exactly once, at the very end:

### Work it by hand: d = 2, two GPUs

Before the real shapes, do one all the way through on numbers you can check.
Hidden size `d = 2`, intermediate `d_ff = 4`, ReLU instead of SiLU, two GPUs.

```
x  = [1  2]                W1 = [ 1  0   2  -1 ]      W2 = [ 1   0 ]
                                [ 0  1  -2   3 ]           [ 0   1 ]
                                                           [ 1   1 ]
                                                           [ 2  -1 ]
```

**One GPU, for reference:**

```
h      = x @ W1  = [1  2  -2  5]
act(h) = relu(h) = [1  2   0  5]        the -2 is clipped
out    = act(h) @ W2 = [11  -3]
```

**Now split `d_ff = 4` down the middle.** GPU 0 takes columns 0–1 of `W1` and
rows 0–1 of `W2`; GPU 1 takes columns 2–3 and rows 2–3.

```
                    GPU 0                          GPU 1

W1 slice      [ 1  0 ]                        [  2  -1 ]
              [ 0  1 ]                        [ -2   3 ]

h  = x @ W1s   [1  2]                          [-2  5]
act(h)         [1  2]                          [ 0  5]     ← relu, locally
                                                             no neighbour needed

W2 slice      [ 1  0 ]                        [ 1   1 ]
              [ 0  1 ]                        [ 2  -1 ]

partial       [1  2]                          [10  -5]
```

Neither partial is the answer — each is full width `(1×2)` but contains only
its own half of the sum. Add them:

```
   [ 1   2 ]  +  [ 10  -5 ]  =  [ 11  -3 ]      ✓ matches the single-GPU result
   └ GPU 0 ┘     └  GPU 1 ┘        └ exact, not approximate ┘
```

Three things to take from the numbers:

- **`x` was replicated.** Both GPUs read the same `[1 2]`; nothing was sent to
  make that happen (it arrived from the previous layer's all-reduce).
- **The activation ran on a half.** GPU 1 clipped its `-2` without ever seeing
  GPU 0's values, because ReLU looks at one element at a time.
- **The partials are full-width, not half-width.** This surprises people: GPU 0
  produces `[1 2]`, a complete-shaped output that is simply *wrong* until GPU 1's
  contribution is added. That is what "partial sum" means, and it is why the
  combine is an **add**, not a concatenate.

---

Draw it with real shapes so the arithmetic is visible. An MLP block is *not*
two square matrices — it widens, then narrows. For Qwen3-0.6B the hidden size
is `d = 1024` and the intermediate width is `d_ff = 3072`, three times wider:

```
      x           W1              h            W2          out
   (1×1024) → (1024×3072) → (1×3072) → (3072×1024) → (1×1024)
                                 ▲
                                 └── "the middle": h, the intermediate
                                     hidden state, d_ff = 3072 wide.
                                     3× wider than the input and output.

   in  1024  ──widen──►  3072  ──narrow──►  1024  out
                          ▲
              this is the dimension TP cuts
```

**`h` is what TP splits** — not `x`, not `out`, but the wide vector between the
two matmuls. Each GPU takes half of it: 1536 of the 3072.

Splitting there is what makes the scheme work, for two reasons:

- **It is the widest thing in the block**, so cutting it removes the most work
  per GPU. Cutting a 1024-wide vector would buy less.
- **It is purely internal.** `x` arrives from the previous layer and `out` goes
  to the next, so both must exist in full on every GPU. `h` exists only inside
  this block — nobody outside needs to see it whole, so no one has to be told
  it was cut.

Take two GPUs.

**First matmul: split `W1` down its columns.** Each GPU holds a `1024 × 1536`
slice — half the intermediate width:

```
GPU 0:  x (1×1024)  ×  W1[:, 0:1536]      ->  h_0 (1×1536)   half the hidden state
GPU 1:  x (1×1024)  ×  W1[:, 1536:3072]   ->  h_1 (1×1536)   the other half
```

A column split needs no communication: each GPU reads the same input row `x`
(replicated — it arrived from the previous layer's all-reduce) but only its own
half of the weights, so both sides compute independently.

**Then the activation — and this is the step that makes the whole scheme work.**
A real MLP applies a nonlinearity between the two matmuls; Qwen3 uses SiLU.

!!! note "Two conventions for the same thing"
    Papers write `out = W2 · act(W1 · x)` — `x` is a *column* vector and the
    weights multiply from the left. Code writes it transposed, because a batch
    of tokens is naturally rows:

    ```
    math:   out = W2 · act(W1 · x)        x is (1024×1), a column
    code:   out = act(x @ W1) @ W2        x is (1×1024), a row
    ```

    Identical operation, mirrored notation. **This lecture uses the code
    convention** — `x @ W1` — because that is what you will type and what the
    shapes above assume.

If the
activation needed the *whole* hidden state, the split would break here and you'd
need a gather before you could continue. It doesn't:

```
   act is ELEMENTWISE — each output depends only on the value beneath it

   h    :  [ h₀ h₁ h₂ ... h₁₅₃₅ │ h₁₅₃₆ ... h₃₀₇₁ ]
             └─ GPU 0 has these ┘ └─ GPU 1 has these ┘
                    ↓ ↓ ↓                  ↓ ↓ ↓
   act(h):  [ σ σ σ  ...  σ     │  σ  ...  σ      ]
             └── computed locally ┘ └── computed locally ┘

   no neighbour needed → no communication
```

Because SiLU acts on each element independently, GPU 0 can apply it to `h_0`
without ever seeing `h_1`. The activation *commutes with the column split*. Had
the middle been something mixing across the width — a softmax over `d_ff`, a
normalization — this would not hold, and that's precisely why attention (which
has a softmax across positions) shards by **head** rather than by hidden width:
each head's softmax is self-contained.

**Second matmul: split `W2` across its rows.** Each GPU holds a `1536 × 1024`
slice, and consumes only the half of the activated hidden state it already has —
no gathering in between:

```
GPU 0:  act(h_0) (1×1536)  ×  W2[0:1536, :]      ->  out_0 (1×1024)  partial sum
GPU 1:  act(h_1) (1×1536)  ×  W2[1536:3072, :]   ->  out_1 (1×1024)  partial sum
                                           all-reduce
                              out = out_0 + out_1    (1×1024)
```

This is the `k`-axis split from the start of the section, now with real widths:
`Σ_k` runs over 0..3071, GPU 0 owns the terms for k = 0..1535 and GPU 1 the rest.
Every term appears exactly once, in exactly one group, so `out_0 + out_1` equals
the full result **algebraically exactly** — the all-reduce is correct, not an
approximation. And that is why a column-then-row pair needs only *one*
all-reduce: the two matmuls hand the split to each other directly, the
elementwise activation passes it through untouched, and only the final output
must be combined.

(Qwen3's MLP is *gated*: it has two up-projections, `gate_proj` and `up_proj`,
multiplied together elementwise before `down_proj`. Both shard column-wise
exactly as `W1` does, and their elementwise product is — again — elementwise, so
the argument is unchanged. One all-reduce, still.)

Column-then-row means **one all-reduce per MLP block**, not two.

**Attention shards the same way, but by heads rather than by width.** The
reason is the one from the activation discussion: attention has a softmax, which
mixes values across positions, so you cannot cut a head in half and still
compute it locally. A whole head, though, is self-contained — its softmax only
ever looks at its own scores. So the unit of sharding is the head:

```
   16 query heads, 8 KV heads (Qwen3-0.6B), across 2 GPUs

   GPU 0:  Q heads 0..7    KV heads 0..3   ─┐
   GPU 1:  Q heads 8..15   KV heads 4..7   ─┤ each computes its heads
                                            │ end to end, alone
                    ↓                       │
            concat along the head dim ──────┘
                    ↓
            output projection (row-split)  →  all-reduce
```

Same shape as the MLP: independent work, then one all-reduce at the output
projection.

> **GQA constrains your TP degree.** GQA — grouped-query attention, where several
> query heads share one key/value head — means the shardable unit is the *KV*
> head, and Qwen3-0.6B has only 8 of those (against 16 query heads).
>
> The constraint is **divisibility, not just a ceiling**: each rank needs a whole
> number of KV heads, so the legal TP degrees are the divisors of 8 — TP-1, 2, 4,
> 8. TP-3 and TP-6 are not options, even though both are below 8.
>
> ```
> TP-2:  8 KV heads / 2 = 4 each   ✓
> TP-4:  8 KV heads / 4 = 2 each   ✓
> TP-8:  8 KV heads / 8 = 1 each   ✓   the ceiling
> TP-16: 8 KV heads / 16 = 0.5     ✗   must replicate KV heads across ranks,
>                                      giving back some of the memory saving
> ```
>
> `d_ff` divisibility is a co-equal constraint on the MLP side: `3072 / TP` must
> also come out whole (it does for all of 1, 2, 4, 8). In practice KV heads bind
> first, because there are far fewer of them.
>
> (To be clear about scale: nobody runs TP-8 on a 0.6B model — the collectives
> would swamp the compute. The *arithmetic* is what transfers; on a 70B model
> with 8 KV heads the same divisibility rule is what caps you at TP-8.) This is
> a real limit people hit, and it's exactly the "where scaling stops" question
> this lecture is about.

So: **two all-reduces per transformer layer.** For a 28-layer model, that's 56
collectives per forward pass (2 × 28). That's the cost, and it's why interconnect matters.

### Why scaling isn't linear

Here is the asymmetry that decides how far TP scales:

```
   add a GPU  ──►  compute per GPU     falls    (÷ N)
              ──►  communication       does NOT
```

Halve the work and you halve the time, but the GPUs still have to talk — and
past some N the talking dominates. This section turns that into a number you can
predict instead of a surprise.

One symbol first: **`S` is the size of the tensor being summed**, in bytes — the
layer's output activation, the thing every all-reduce has to reconcile. All the
cost below is measured in multiples of `S`.

The standard implementation is a **ring all-reduce**: the GPUs stand in a
circle, each holding `S/N` of the data. Each passes its share one step around
the ring, then a second lap spreads the total back.

Draw the circle with 4 GPUs, each starting with its own partial sum split into
4 chunks:

```
                    ┌─────────┐
              ┌────>│  GPU 0  │────┐
              │     └─────────┘    │            lap 1 (reduce-scatter):
              │                    ▼            each GPU adds the chunk it
         ┌─────────┐          ┌─────────┐       receives to its own, then
         │  GPU 3  │          │  GPU 1  │       passes it on. After N−1
         └─────────┘          └─────────┘       hops, each GPU owns the
              ▲                    │            complete sum of ONE chunk.
              │     ┌─────────┐    │
              └─────│  GPU 2  │<───┘            lap 2 (all-gather):
                    └─────────┘                 those finished chunks go
                                                round again so everyone
      each hop carries S/N bytes                ends up with all of them.
```

Every lap has `N−1` hops, so every GPU sends `N−1` chunks of `S/N` bytes and
receives the same, in both directions:

```
bytes per rank  =  (N−1 + N−1) × S/N      two laps of N−1 hops each
                =  2(N−1)/N × S
```

A ring all-reduce moves `2(N-1)/N · S` bytes per rank, where a rank is one
GPU's identity in the group, which *approaches* a constant `2S`
rather than falling like `compute/N`. Plug in the GPU counts:

```
N = 2:    2 × 1/2 × S  = 1.00 × S
N = 4:    2 × 3/4 × S  = 1.50 × S
N = 8:    2 × 7/8 × S  = 1.75 × S
N = 16:   2 × 15/16 × S = 1.875 × S   ->  approaches 2S, never reaches it
```

**Put a number on `S`.** For this book's model, one all-reduce carries the
layer's output activation: `batch × d × 2 bytes`, with `d = 1024` in fp16.

```
   batch  1:  S = 1 × 1024 × 2  =    2 KiB
   batch 32:  S = 32 × 1024 × 2 =   64 KiB

   wire time for 2S, all 56 collectives of a forward pass:

                       batch 1        batch 32
   NVLink 4 (900 GB/s)   ~0.3 µs        ~8 µs
   PCIe 5   (128 GB/s)   ~1.8 µs       ~57 µs
```

Which is a genuinely surprising answer: **the bytes are nothing.** Even on PCIe,
even summed over all 56 collectives, you are spending microseconds on a decode
step that takes milliseconds.

So on a small model TP is not limited by *bandwidth* at all — it is limited by
the **fixed cost of each collective**: kernel launch, synchronization, and the
round-trip in which every GPU must wait for the slowest. Fifty-six of those per
forward pass, each a barrier where all GPUs stop and agree, is what actually
eats the win. The bandwidth table below matters for large models and long
sequences, where `S` grows until the bytes do dominate; below that, latency
rules and the table will mislead you.

Each doubling of GPUs halves the compute on each GPU, but the all-reduce bytes
edge merely closer to a floor of `2S`. The gap between the two curves is where
the scaling stops:

```
time = compute/N + communication(N)
```

Past some N, communication dominates and adding GPUs stops helping. Where that
happens depends on interconnect, the wires that connect the GPUs:

| Link | Bandwidth per GPU |
|---|---|
| NVLink 4 (Hopper) | 900 GB/s |
| NVLink 5 (Blackwell) | 1,800 GB/s |
| NVLink 6 (Rubin, announced) | 3,600 GB/s |
| **PCIe 5 ×16** | **~64 GB/s each way (~128 bidirectional)** |

*(NVIDIA's published figures as of 2026; NVLink numbers are bidirectional per
GPU. Rubin is announced, not shipping, check before planning around it.)*

**Roughly an order of magnitude — 900 / 64 ≈ 14× — in the thing that isn't
parallelized**: and the gap has widened with each NVLink generation, not
narrowed. This is why the same
model scales beautifully on one box and poorly on another with identical GPUs.

Note also what this means for **rented** hardware: a Vast.ai listing advertising
"2× 3090" tells you nothing about the link between them. Check
`nvidia-smi topo -m` before you interpret a scaling curve.

### Where the rules of thumb break

Conventional guidance says: low interconnect bandwidth → use pipeline parallelism
instead of TP.

The [field notes](field-notes.md) record an operator with **2× GH200 and no
NVLink** (PCIe only, they quote 125 GB/s against NVLink's 900) who followed
exactly that advice.
**Pipeline parallel lost. TP2 won.** On the hardware profile where guides say it
shouldn't.

The same operator found `--max-num-seqs 16`, a scheduler concurrency limit from
Lecture 08 (the cap on how many sequences share the GPU at once), mattered more
than the parallelism strategy did.

Take the lesson generally: **rules of thumb about parallelism are workload- and
model-dependent.** Measure your own scaling curve. That's the actual deliverable
of this lecture.

---

## Do it twice

### First: declaratively, in JAX

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(jax.devices(), axis_names=("tp",))

# column-parallel: shard the output dimension
w1 = jax.device_put(w1, NamedSharding(mesh, P(None, "tp")))
# row-parallel: shard the input dimension
w2 = jax.device_put(w2, NamedSharding(mesh, P("tp", None)))
```

That's the whole thing. **You annotated a layout; you never wrote a collective.**
XLA derives that a row-parallel matmul followed by a use of the full output
requires an all-reduce, and inserts it.

Dump the HLO and **find the `all-reduce` XLA inserted.** Seeing a collective you
did not write, at exactly the position the math requires, is the moment this
lecture is built around.

### Then: by hand, in PyTorch

```python
class ColumnParallelLinear(nn.Module):
    def forward(self, x):
        return F.linear(x, self.weight_shard)   # no comms

class RowParallelLinear(nn.Module):
    def forward(self, x):
        out = F.linear(x, self.weight_shard)    # partial sum
        dist.all_reduce(out)                    # <- you write this
        return out
```

Now you're placing the collective yourself, in the position XLA chose for you.
Compare against nano-vllm's `layers/linear.py`, which does exactly this.

---

## Build it

1. Shard the JAX model with `NamedSharding`. Verify output matches single-device.
2. Find the inserted all-reduce in the HLO.
3. Implement `ColumnParallelLinear` / `RowParallelLinear` in PyTorch with NCCL.
4. Verify TP output matches single-GPU output exactly.
5. **Measure the scaling curve** at 1, 2, 4 GPUs, plot with
   `bench/plot.py::scaling`, which takes measured throughput against ideal linear.
6. Find where it goes sublinear, and check `nvidia-smi topo -m` to see whether
   you have NVLink or PCIe.

**Predict first:** what speedup at 2 GPUs? At 4? Write both down before measuring.

---

## What you should see

**Sublinear scaling.** 2 GPUs give well under 2×. The gap is communication.

**Better scaling with NVLink** than PCIe, on identical GPUs.

**Better scaling on larger models**: more compute per layer amortizes the same
all-reduce.

**Worse scaling at small batch sizes**: less compute to hide the collective
behind.

**Latency improves; throughput per GPU falls.** You're spending hardware to make
one user faster. That's the trade, and it's the right one only sometimes.

---

## Go deeper

- **[Megatron-LM](https://arxiv.org/abs/1909.08053)** (Shoeybi et al.), §3 has
  the column/row split you just implemented. The original.
- **Kiely §5.4–5.4.1** (p.142–145), TP/PP/EP compared, and TP for latency.
- **Kiely §5.4.3** (p.146), multi-node, where TP stops being the answer.
- **nano-vllm `nanovllm/layers/linear.py`**: a readable production TP
  implementation.
- **[Field notes](field-notes.md)**: the GH200 case where PP lost to TP without
  NVLink.

---

## Check yourself

1. Why is TP a latency optimization when batching is a throughput optimization?
2. Why does column-then-row need only one all-reduce per MLP?
3. Your 2-GPU speedup was 1.6×. Where did 0.4 go, and how would you confirm it?
4. Same GPUs, one box with NVLink and one without. Why do they scale differently?
5. An operator without NVLink found TP beat PP, contradicting the guides. What
   does that tell you about applying rules of thumb to your own setup?

---

## Next

**[23. MoE and expert parallelism](23-moe-and-expert-parallelism.md)**: a
different axis to split along, and how most frontier open models are now built.

The distinction to get exactly right: **total vs. active parameters.**
DeepSeek-V3 is 671B total and 37B active, and you must hold all 671B in VRAM,
because the router might pick any expert.

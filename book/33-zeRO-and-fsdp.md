# 33. Data parallelism, ZeRO-1/2/3, and FSDP

**Demo:** `book/code/zero_memory.py` · **Build:** `train/zero.py` · **Test:** `tests/test_33_zero.py`
**Moves:** per-GPU model-state memory down by up to $D\times$ across $D$ GPUs
**Prereq:** [30. The training ledger](30-training-arithmetic.md), [22. Tensor parallelism](22-tensor-parallelism.md)

---

## The problem

You rent an $8 \times \text{RTX 3090}$ node ($8 \times 24\text{ GiB} = 192\text{ GiB}$
of aggregate VRAM) to train a **7B dense model** ($\Psi = 7 \times 10^9$).

From Lecture 30's ledger, mixed-precision AdamW state for a 7B model is:

$$16 \Psi = 16 \times 7 \times 10^9\text{ B} = 112\text{ GB} = 104.31\text{ GiB}$$

Look at the two numbers:
- Your 8 GPUs have **192 GiB** of VRAM combined — almost twice what the 7B
  model's state needs ($104.31\text{ GiB}$).
- Yet standard **Distributed Data Parallel (DDP)** crashes immediately on all 8
  GPUs, because DDP puts a **complete 104.31 GiB copy** of the weights,
  gradients, and optimizer state on **every single GPU**.

Across 8 GPUs, DDP spends $8 \times 104.31 = 834.5\text{ GiB}$ to store
$104.31\text{ GiB}$ of information. Seven-eighths of your cluster's memory is
holding duplicate copies of the exact same numbers.

**ZeRO (Zero Redundancy Optimizer, Rajbhandari et al., 2019)** and PyTorch's
**FSDP (Fully Sharded Data Parallel)** ask one question: *why replicate state
across $D$ data-parallel GPUs when each GPU can hold $1/D$ of it and reconstruct
the rest on the fly?*

---

## See it

```bash
uv run python book/code/zero_memory.py
```

```
Qwen3-0.6B (0.60B params) across 8 GPUs
=========================================================================
  stage          per-GPU state     reduction     comm / GPU / step
  DDP (ZeRO-0)        8.88 GiB       1.00x            1.94 GiB
  ZeRO-1 (opt)        3.05 GiB       2.91x            1.94 GiB
  ZeRO-2 (+grad)      2.08 GiB       4.27x            1.94 GiB
  ZeRO-3 / FSDP       1.11 GiB       8.00x            2.91 GiB

7B dense model (7.00B params) across 8 GPUs
=========================================================================
  stage          per-GPU state     reduction     comm / GPU / step
  DDP (ZeRO-0)      104.31 GiB       1.00x           22.82 GiB
  ZeRO-1 (opt)       35.86 GiB       2.91x           22.82 GiB
  ZeRO-2 (+grad)     24.45 GiB       4.27x           22.82 GiB
  ZeRO-3 / FSDP      13.04 GiB       8.00x           34.23 GiB
```

Look at the two columns side by side:
1. **ZeRO-1 and ZeRO-2 cut model-state memory by $2.91\times$ and $4.27\times$
   while moving the EXACT same network bytes ($1.94\text{ GiB}$ for Qwen3-0.6B,
   $22.82\text{ GiB}$ for 7B) as plain DDP!**
2. **ZeRO-3 / FSDP cuts model-state memory by the full $8.00\times$** ($104.31 \to 13.04\text{ GiB}$,
   fitting the 7B model comfortably on 24 GiB 3090s!) at a cost of **exactly
   $1.5\times$ the network traffic** ($22.82 \to 34.23\text{ GiB}$).

Both facts come out of one collective identity. Let's derive them.

---

## The idea

### 1. The collective identity: one `all-reduce` IS one `reduce-scatter` plus one `all-gather`

In Lecture 22 (Tensor parallelism), we used `all-reduce` as a black box: $D$
GPUs each hold a vector of size $M$ bytes ($M = 2\Psi$ for BF16 gradients), and
at the end every GPU holds the elementwise sum.

How does a ring or tree `all-reduce` actually move those $M$ bytes across $D$
GPUs? In **two halves**:

```
1. REDUCE-SCATTER (costs ((D-1)/D) * M bytes per GPU)
   Before: GPU 0 has [g0_0, g0_1]     GPU 1 has [g1_0, g1_1]
   After:  GPU 0 has [g0_0 + g1_0]    GPU 1 has [g0_1 + g1_1]
   (Each GPU now owns the FULLY SUMMED gradient for its 1/D shard!)

2. ALL-GATHER (costs ((D-1)/D) * M bytes per GPU)
   Before: GPU 0 has [w0_new]         GPU 1 has [w1_new]
   After:  GPU 0 has [w0_new, w1_new] GPU 1 has [w0_new, w1_new]
   (Every GPU now has the full updated tensor again!)
```

Count the bytes sent per GPU in each half:
- In `reduce-scatter`, the vector is split into $D$ chunks of size $M / D$, and
  each GPU sends $D - 1$ chunks around the ring:
  $$\text{Bytes}_{\text{RS}} = \frac{D - 1}{D} M$$
- In `all-gather`, each GPU broadcasts its $M / D$ chunk to the other $D - 1$
  steps around the ring:
  $$\text{Bytes}_{\text{AG}} = \frac{D - 1}{D} M$$
- Sum of the two halves:
  $$\text{Bytes}_{\text{AllReduce}} = \text{Bytes}_{\text{RS}} + \text{Bytes}_{\text{AG}} = 2 \frac{D - 1}{D} M$$

With $M = 2\Psi$ bytes (BF16 gradients), in the large-$D$ limit
($\frac{D-1}{D} \approx 1$), an `all-reduce` moves:

$$\text{Bytes}_{\text{DDP}} = 2 \frac{D - 1}{D} (2\Psi) \approx 4\Psi \text{ bytes per GPU per step}$$

At $D = 8$ GPUs:
- For Qwen3-0.6B ($\Psi = 596.05\text{M}$): $4 \times \frac{7}{8} \times 596{,}049{,}920 = 2{,}086{,}174{,}720\text{ B} = 1.94\text{ GiB}$.
- For 7B ($\Psi = 7\text{B}$): $4 \times \frac{7}{8} \times 7 \times 10^9 = 24.5\text{ GB} = 22.82\text{ GiB}$.

---

### 2. Deriving ZeRO-1, ZeRO-2, and ZeRO-3 / FSDP term by term

Recall Lecture 30's $16\Psi$ breakdown per GPU:

$$\text{State}_{\text{DDP}} = \underbrace{2\Psi}_{\text{BF16 weights}} + \underbrace{2\Psi}_{\text{BF16 grads}} + \underbrace{12\Psi}_{\text{FP32 master } w + m + v} = 16\Psi$$

Now watch what happens as we shard those three pieces across $D$ data-parallel
GPUs:

```
                 DDP (ZeRO-0)      ZeRO-1            ZeRO-2            ZeRO-3 / FSDP
                ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
  Weights (2Ψ)  │  full 2Ψ   │    │  full 2Ψ   │    │  full 2Ψ   │    │   2Ψ / D   │
                ├────────────┤    ├────────────┤    ├────────────┤    ├────────────┤
  Grads   (2Ψ)  │  full 2Ψ   │    │  full 2Ψ   │    │   2Ψ / D   │    │   2Ψ / D   │
                ├────────────┤    ├────────────┤    ├────────────┤    ├────────────┤
  Opt    (12Ψ)  │ full 12Ψ   │    │  12Ψ / D   │    │  12Ψ / D   │    │  12Ψ / D   │
                └────────────┘    └────────────┘    └────────────┘    └────────────┘
```

#### Stage 1: ZeRO-1 (shard optimizer states only, $12\Psi / D$)

- **Memory per GPU:**
  $$\text{Bytes}_{\text{ZeRO-1}} = 2\Psi + 2\Psi + \frac{12\Psi}{D}$$
  At $D = 8$: $4\Psi + 1.5\Psi = 5.5\Psi$, a **$16 / 5.5 = 2.91\times$ reduction**
  ($8.88 \to 3.05\text{ GiB}$ on Qwen3-0.6B; $104.31 \to 35.86\text{ GiB}$ on 7B).
- **How the step works:**
  1. Forward and backward run exactly like DDP (every GPU already has full $2\Psi$ weights).
  2. Instead of a full `all-reduce` on gradients, run **only the first half**:
     `reduce-scatter` on the $2\Psi$ gradients ($\approx 2\Psi$ bytes moved). Now
     GPU $r$ holds the summed gradient for parameter shard $r$.
  3. GPU $r$ runs `optimizer.step()` on **only its $1/D$ shard** of FP32 master
     weights, $m$, and $v$, producing updated BF16 weights for shard $r$.
  4. Run **the second half**: `all-gather` the updated BF16 weight shards across
     all GPUs ($\approx 2\Psi$ bytes moved).
- **Total communication:**
  $$\underbrace{2\Psi}_{\text{grad reduce-scatter}} + \underbrace{2\Psi}_{\text{weight all-gather}} = 4\Psi \quad (\text{identical to DDP!})$$

??? question "Wait — how can ZeRO-1 save 65% of model-state memory with ZERO extra network traffic?"
    Because DDP was doing redundant work! In DDP, `all-reduce` is a
    `reduce-scatter` followed by an `all-gather` of gradients, and then all $D$
    GPUs execute the **exact same AdamW update** on all $\Psi$ parameters in
    lockstep. ZeRO-1 simply moves the `optimizer.step()` **between** the
    `reduce-scatter` and the `all-gather`: you `reduce-scatter` gradients, each
    GPU updates its own $1/D$ slice of AdamW state, and then you `all-gather`
    the updated BF16 weights. Same two collective halves, same $4\Psi$ bytes on
    the wire, $D\times$ less optimizer memory and $D\times$ less optimizer compute!
    [Full answer](qa.md#how-can-zero-1-save-memory-with-zero-extra-network-traffic)

#### Stage 2: ZeRO-2 (shard optimizer states + gradients, $(2\Psi + 12\Psi)/D$)

Why keep the full $2\Psi$ gradient tensor on every GPU after layer $\ell$'s
`reduce-scatter` has already finished?
- As soon as layer $\ell$ computes $\nabla_{W_\ell}$ during backward, immediately
  `reduce-scatter` it so GPU $r$ keeps only its $1/D$ slice of $\nabla_{W_\ell}$
  and **frees the other $(D-1)/D$ slices** before layer $\ell-1$'s backward pass!
- **Memory per GPU:**
  $$\text{Bytes}_{\text{ZeRO-2}} = 2\Psi + \frac{2\Psi + 12\Psi}{D}$$
  At $D = 8$: $2\Psi + 1.75\Psi = 3.75\Psi$, a **$16 / 3.75 = 4.27\times$ reduction**
  ($8.88 \to 2.08\text{ GiB}$ on Qwen3-0.6B; $104.31 \to 24.45\text{ GiB}$ on 7B).
- **Total communication:** still **one `reduce-scatter` + one `all-gather` = $4\Psi$ bytes**!

#### Stage 3: ZeRO-3 / PyTorch FSDP (shard everything, $16\Psi / D$)

Now shard the $2\Psi$ BF16 weights too. Each GPU holds only $1/D$ of every
layer's weights at rest:
- **Memory per GPU:**
  $$\text{Bytes}_{\text{ZeRO-3}} = \frac{2\Psi + 2\Psi + 12\Psi}{D} = \frac{16\Psi}{D} \quad (+\text{ one live layer buffer})$$
  At $D = 8$: $16\Psi / 8 = 2\Psi$, an **$8.00\times$ reduction**
  ($104.31 \to 13.04\text{ GiB}$ on 7B).
- **How the step works (and where the extra $2\Psi$ traffic comes from):**
  1. **Forward pass, layer $\ell$:** `all-gather` layer $\ell$'s BF16 weights
     from all $D$ GPUs, run forward $Y_\ell = X_\ell W_\ell$, then **immediately
     free** the gathered weights you don't own! Across all layers: $+2\Psi$ bytes.
  2. **Backward pass, layer $\ell$:** you threw away the non-local weights after
     forward, so you must **`all-gather` layer $\ell$'s BF16 weights a second
     time** ($+2\Psi$ bytes), compute $\nabla_X$ and $\nabla_W$, free the
     non-local weights, and **`reduce-scatter` the gradients $\nabla_W$**
     ($+2\Psi$ bytes).
  3. **Optimizer step:** each GPU updates its $1/D$ shard in place (no weight
     `all-gather` needed at the end because step 1 of the next iteration will
     gather them per layer!).
- **Total communication:**
  $$\underbrace{2\Psi}_{\text{fwd weight AG}} + \underbrace{2\Psi}_{\text{bwd weight AG}} + \underbrace{2\Psi}_{\text{bwd grad RS}} = 6\Psi \text{ bytes} \quad (1.5\times \text{ DDP})$$

??? question "Why does DeepSeek-V3 use ZeRO-1 instead of ZeRO-3 / FSDP at 2,048 GPUs?"
    Because ZeRO-3's two weight `all-gather`s sit directly on the forward and
    backward path of **every micro-batch**, whereas ZeRO-1 communicates only
    **once per optimizer step** (after accumulating across all $K$ micro-batches!).
    With $K$ gradient-accumulation micro-batches, ZeRO-1's communication per step
    is $4\Psi$, while ZeRO-3's is $(4K + 2)\Psi$ — at $K = 8$, ZeRO-3 moves
    $34\Psi / 4\Psi = 8.5\times$ more bytes across the network! When pipeline and
    expert parallelism already fit the weights in VRAM, ZeRO-1 gives the $75\%$
    optimizer-memory savings without paying per-microbatch weight gathers.
    [Full answer](qa.md#why-does-deepseek-v3-use-zero-1-instead-of-zero-3-at-scale)

---

## The code

```python
def zero_state_bytes(params: int, dp_degree: int, stage: int) -> int:
    w, g, opt = 2 * params, 2 * params, 12 * params
    if stage == 0:
        return w + g + opt
    if stage == 1:
        return w + g + opt // dp_degree
    if stage == 2:
        return w + (g + opt) // dp_degree
    if stage == 3:
        return (w + g + opt) // dp_degree
    raise ValueError(f"Unknown ZeRO stage: {stage}")


def comm_bytes_per_step(params: int, stage: int, param_bytes: int = 2) -> int:
    mult = 2 if stage in (0, 1, 2) else 3
    return mult * param_bytes * params
```

---

## Build it

1. Implement `zero_state_bytes` and `comm_bytes_per_step` in `train/zero.py`.
2. Run the tests:

```bash
uv run pytest tests/test_33_zero.py -v
```

3. Compute the minimum number of 24 GiB RTX 3090s required to hold the model
   states of a **14B parameter model** under DDP, ZeRO-1, ZeRO-2, and ZeRO-3
   (leaving at least 6 GiB per GPU for activations).

---

## What you should see

- **ZeRO-1 and ZeRO-2 are strictly free in network volume** compared to DDP at
  $K=1$ ($4\Psi$ bytes), while cutting state memory by $2.91\times$ and
  $4.27\times$ at $D=8$.
- **ZeRO-3 / FSDP scales state memory linearly as $16\Psi / D$**, trading $1.5\times$
  communication (or $(2K+1)/2\times$ under gradient accumulation) for the ability
  to fit models larger than a single GPU.

---

## Go deeper

- **[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)** (Rajbhandari et al., 2019) — Sections 3 and 5 derive the exact memory and communication bounds above.
- **[PyTorch FSDP2 (`fully_shard`)](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)** — DTensor-based per-parameter sharding with overlapped prefetching.

---

## Check yourself

1. Decompose a ring `all-reduce` of $2\Psi$ bytes into two collective primitives
   and state how many bytes each moves.
2. Why does ZeRO-1 move the exact same number of bytes per step as DDP?
3. You use $K = 16$ gradient accumulation micro-batches per optimizer step. How
   many bytes per step does ZeRO-1 communicate vs ZeRO-3 (without weight caching
   across micro-batches)?

---

## Next

**[34. TP + SP, Pipeline schedules (1F1B to DualPipe), and 3D parallelism](34-3d-parallelism-and-schedules.md)**:
sharding state across DP ranks doesn't shrink a giant single-layer activation or
hide cross-node latency. Now combine Tensor + Sequence Parallelism inside the
node and Pipeline Parallelism across nodes.

```bash
uv run python book/code/pipeline_bubble.py
```

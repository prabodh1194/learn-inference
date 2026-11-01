# 34. TP + SP, Pipeline schedules (1F1B to DualPipe), and 3D parallelism

**Demo:** `book/code/pipeline_bubble.py` · **Build:** `train/pipeline.py` · **Test:** `tests/test_34_pipeline.py`
**Moves:** shrinks per-rank activation memory by $t\times$ at zero extra network cost; cuts pipeline bubbles from $\frac{p-1}{m+p-1}$ toward zero
**Prereq:** [22. Tensor parallelism](22-tensor-parallelism.md), [31. Activations and checkpointing](31-activation-memory.md), [33. ZeRO and FSDP](33-zeRO-and-fsdp.md)

---

## The problem

In Lecture 22 (Tensor parallelism), we split each layer's weight matrices over
$t$ GPUs using Column-Parallel followed by Row-Parallel linear layers, linked by
one `all-reduce` per sub-block. For inference, that was the whole story.

In training, two new problems appear. First, **plain Tensor Parallelism leaves
$10 s b h$ of activations per layer completely unsharded**. Norms, residuals, and
the inputs to the column-parallel projections sit *outside* the tensor-parallel
regions, so every TP rank holds an identical copy of them. At $t = 8$, those
replicated buffers take **77% of your remaining activation memory**. Second,
**Pipeline Parallelism (PP) introduces idle bubbles and an in-flight activation
explosion**. In inference, forward-only micro-batches stream down pipeline
stages without stopping. In training, all $m$ micro-batches must finish both
forward and backward before `optimizer.step()`, so GPUs sit idle while the
pipeline fills and drains.

---

## See it

```bash
uv run python book/code/pipeline_bubble.py
```

```
Per-layer activation factor s*b*h*(10 + 24/t) vs s*b*h*(34/t)
=========================================================================
  TP (t)    TP only (no SP)    TP + SP      SP memory reduction   extra comm
       1         34.00 sbh      34.00 sbh          1.00x            0 bytes
       2         22.00 sbh      17.00 sbh          1.29x            0 bytes
       4         16.00 sbh       8.50 sbh          1.88x            0 bytes
       8         13.00 sbh       4.25 sbh          3.06x            0 bytes

Pipeline Parallelism (p=4 stages): bubble fraction and peak in-flight micro-batches
=========================================================================
  micro-batches (m)   bubble (p-1)/(m+p-1)   GPipe in-flight   1F1B in-flight
                  4            42.9%                  4x                4x
                  8            27.3%                  8x                4x
                 16            15.8%                 16x                4x
                 32             8.6%                 32x                4x
                 64             4.5%                 64x                4x
```

Look at the two tables:
- At $t = 8$, adding **Sequence Parallelism (SP)** to Tensor Parallelism cuts
  per-layer activation memory from $13.00\,sbh$ to $4.25\,sbh$ (**$3.06\times$
  smaller**) while adding **0 bytes** of network traffic.
- Pushing micro-batches from $m = 4$ to $m = 64$ shrinks the 4-stage pipeline
  bubble from **42.9% to 4.5%**, and **1F1B** caps peak in-flight activation
  memory at **$4\times$** instead of GPipe's **$64\times$**.

---

## The idea

### 1. Tensor Parallelism + Sequence Parallelism (TP + SP): $3.06\times$ less activation memory for zero extra bytes

Recall from Korthikanti et al. (2022) that a standard transformer layer's
non-score activation memory ($34 s b h$ bytes in their GPT formulation, or
$38 s b h$ for Qwen3's GQA+SwiGLU in Lecture 31) splits into two regions. Inside
the TP regions ($24 s b h$), the $Q, K, V$ outputs, attention output projection
input, and MLP intermediate tensors are split along the head or hidden axis over
the $t$ TP ranks, taking $\frac{24}{t} s b h$ bytes per rank. Outside the TP
regions ($10 s b h$), the two LayerNorm/RMSNorm inputs ($4 s b h$), the inputs
to the QKV and MLP blocks ($4 s b h$), and residual/dropout buffers ($2 s b h$)
are consumed after the TP `all-reduce`, so they stay **replicated on all $t$
GPUs**:

$$\text{Act}_{\text{TP only}} = s b h \left(10 + \frac{24}{t}\right)$$

At $t = 8$, $\frac{24}{8} = 3$, so the sharded part is $3 s b h$ while the
replicated part is still $10 s b h$ ($10 / 13 = 76.9\%$ of the total!).

Now use the collective identity from Lecture 33: **one `all-reduce` equals one
`reduce-scatter` plus one `all-gather`**.

Notice what RMSNorm and residual addition actually compute: they operate
**independently on each token along the sequence dimension $s$**! They do not
need all $s$ tokens on the same GPU; they only need all $h$ hidden channels of
whichever $s / t$ tokens they hold.

So instead of ending the Row-Parallel linear layer with an `all-reduce` (which
leaves every GPU with all $s$ tokens), end it with a **`reduce-scatter` along
the sequence dimension $s$** ($\bar{g}$):

```
PLAIN TP (replicates [s, b, h] across all t ranks, 1 all-reduce per sub-block):
  [s, b, h] ──► RMSNorm ──► [s, b, h] ──► ColParallel ──► [s, b, h/t] ──► RowParallel ──(AllReduce)──► [s, b, h]
  (saved: 2sbh)             (saved: 2sbh)                 (sharded /t)

TP + SEQUENCE PARALLELISM (shards sequence s/t outside TP, 1 RS + 1 AG = same bytes!):
  [s/t, b, h] ──► RMSNorm ──► [s/t, b, h] ──(AllGather g)──► [s, b, h] ──► ColParallel ──► [s, b, h/t] ──► RowParallel ──(ReduceScatter ḡ)──► [s/t, b, h]
  (saved: 2sbh/t)             (saved: 2sbh/t — re-gather X in bwd!)
```

Count the bytes saved and the bytes communicated. Every norm and residual buffer
outside the TP block now has shape $(s/t, b, h)$ instead of $(s, b, h)$, so the
$10 s b h$ term divides by $t$:

$$\text{Act}_{\text{TP+SP}} = s b h \left(\frac{10}{t} + \frac{24}{t}\right) = \frac{34}{t} s b h$$

In forward, each sub-block runs one `all-gather` ($g$) before Column-Parallel
and one `reduce-scatter` ($\bar{g}$) after Row-Parallel. Together, one
`all-gather` plus one `reduce-scatter` move the **exact same number of bytes as
the single `all-reduce` that plain TP was already doing**.

??? question "Wait, if the `all-gather` rebuilds the full `[s, b, h]` tensor before the Column-Parallel matmul, doesn't $\nabla_W = X^\top \nabla_Y$ still require saving that full `[s, b, h]` tensor?"
    Only if you keep the gathered copy. Megatron-LM's trick is to **discard**
    the gathered `[s, b, h]` buffer right after the forward matmul finishes and
    keep only the local `[s/t, b, h]` slice in VRAM. During the backward pass,
    an asynchronous `all-gather` rebuilds `[s, b, h]` in a temporary buffer
    right before $\nabla_W = X^\top \nabla_Y$ runs, overlapped underneath the
    $\nabla_X$ computation.
    [Full answer](qa.md#how-does-sequence-parallelism-avoid-saving-the-all-gathered-input-for-backward)

---

### 2. Pipeline Parallelism schedules: GPipe, 1F1B, and Zero-Bubble / DualPipe

Between nodes connected by slower scale-out links (InfiniBand or RoCE), Tensor
Parallelism's per-layer collectives become too chatty (Lecture 22). Instead,
**Pipeline Parallelism (PP)** places whole layers on different stages
$0, 1, \dots, p-1$ and sends only the layer-boundary activation $(s, b, h)$
point-to-point between adjacent stages.

Split the batch into $m$ micro-batches. Let one micro-batch take time $T_F$ for
forward on one stage and $T_B$ for backward on one stage.

#### Schedule 1: GPipe (all forwards, then all backwards)

```
stage 3:              [F0][F1][F2][F3]  [B3][B2][B1][B0]
stage 2:          [F0][F1][F2][F3]          [B3][B2][B1][B0]
stage 1:      [F0][F1][F2][F3]                  [B3][B2][B1][B0]
stage 0:  [F0][F1][F2][F3]                          [B3][B2][B1][B0]
          |<-- fill -->|                            |<-- drain -->|
```

Count the useful time versus the idle bubble time on any stage. Each stage does
$m$ forwards and $m$ backwards of useful work:

$$T_{\text{useful}} = m (T_F + T_B)$$

Meanwhile, a stage waits $(p - 1)$ micro-steps for the pipeline to fill and
$(p - 1)$ micro-steps for the pipeline to drain:

$$T_{\text{bubble}} = (p - 1)(T_F + T_B)$$

Divide the bubble time by the total step time, and cancel $T_F + T_B$ from
numerator and denominator:

$$\text{Bubble fraction} = \frac{T_{\text{bubble}}}{T_{\text{useful}} + T_{\text{bubble}}} = \frac{(p - 1)(T_F + T_B)}{m(T_F + T_B) + (p - 1)(T_F + T_B)} = \frac{p - 1}{m + p - 1}$$

To make $\frac{p-1}{m+p-1}$ small, you must make $m \gg p$ (for instance,
$m = 64$ at $p = 4$ gives $\frac{3}{67} = 4.5\%$). Here is GPipe's catch: Stage
0 runs all $m = 64$ forward passes before `B0` arrives back from Stage 1, so
Stage 0 must hold **all $m = 64$ micro-batches of forward activations in VRAM at
the same time**.

#### Schedule 2: 1F1B (One-Forward-One-Backward, PipeDream-Flush)

Why run `F4` on Stage 0 when `B0` is already ready to run? Once the pipeline
fills ($p$ warmup forwards on Stage 0), alternate **one backward right after
each forward**:

```
stage 3:              [F0][B0][F1][B1][F2][B2][F3][B3]
stage 2:          [F0][F1][B0][F2][B1][F3][B2]    [B3]
stage 1:      [F0][F1][F2][B0][F3][B1]    [B2]    [B3]
stage 0:  [F0][F1][F2][F3][B0]    [B1]    [B2]    [B3]
          |<-- p=4 warmup -->|  |<-- 1F1B steady -->|
```

Look at what happens on Stage 0 as soon as `B0` finishes: **micro-batch 0's
activations are freed before `F4` runs**. Because the total work and the warmup
and drain periods are unchanged, the bubble fraction stays at
$\frac{p - 1}{m + p - 1}$. However, peak in-flight micro-batches on Stage 0 are
now capped at **$\min(p, m)$** instead of $m$. At $p = 4, m = 64$, 1F1B holds
**$4$ micro-batches of activations instead of $64$**, a **$16\times$ reduction
in activation memory** at the exact same throughput.

#### Schedule 3: Zero-Bubble PP and DeepSeek-V3's DualPipe

Recall Lecture 30's backward split. The input gradient $B_x$
($\nabla_X = \nabla_Y W^\top$, $2ND$ FLOPs) must go to stage $\ell - 1$ right
away so the upstream stage can keep working. By contrast, the weight gradient
$B_w$ ($\nabla_W = X^\top \nabla_Y$, $2ND$ FLOPs) is **not needed by any other
stage**; it only updates local weights at the end of the step.

**Zero-Bubble PP** (Qi et al., 2024) runs $B_x$ right away so the upstream stage
never stalls, and holds $B_w$ in reserve to **plug the idle bubble slots**
during warmup and drain.

**DeepSeek-V3's DualPipe** goes one step further for cross-node MoE training. It
feeds two streams of micro-batches from **both ends of the pipeline at once**
(Stage $0 \to p-1$ and Stage $p-1 \to 0$) and overlaps the cross-node MoE
`all-to-all` dispatch and combine of micro-batch $A$ underneath the attention and
$B_w$ compute of micro-batch $B$, hiding the network traffic behind math.

---

### 3. The 3D / 4D Parallelism map: which dimension goes on which wire

Put Lectures 22, 23, 33, and 34 on one hardware map:

| Parallelism | Splits what | Collective | Frequency | Where it lives |
|---|---|---|---|---|
| **TP + SP** | intra-layer matrices & sequence | `all-gather` + `reduce-scatter` | 4 per layer | **intra-node NVLink** ($t \le 8$) |
| **EP (MoE)** | routed experts | `all-to-all` | 4 per MoE layer | intra-node or DualPipe-overlapped IB |
| **PP** | layers across stages | point-to-point `send`/`recv` | 2 per stage boundary | **inter-node** (tiny payload: $2sbh$ bytes) |
| **DP / ZeRO-1** | batches + optimizer state | `reduce-scatter` + `all-gather` | 1 per optimizer step | **outermost ring** across all replicas |

---

## The code

```python
def sp_activation_bytes(
    replicated_bytes: int, tp_sharded_bytes: int, tp_degree: int, use_sp: bool
) -> int:
    if use_sp:
        return (replicated_bytes + tp_sharded_bytes) // tp_degree
    return replicated_bytes + (tp_sharded_bytes // tp_degree)


def pipeline_bubble_fraction(pp_degree: int, n_microbatches: int) -> float:
    return (pp_degree - 1) / (n_microbatches + pp_degree - 1)


def inflight_microbatches(
    pp_degree: int, n_microbatches: int, schedule: str = "1f1b"
) -> int:
    if schedule == "gpipe":
        return n_microbatches
    if schedule == "1f1b":
        return min(pp_degree, n_microbatches)
    raise ValueError(f"Unknown schedule: {schedule}")
```

---

## Build it

1. Implement `sp_activation_bytes`, `pipeline_bubble_fraction`, and
   `inflight_microbatches` in `train/pipeline.py`.
2. Run the tests:

```bash
uv run pytest tests/test_34_pipeline.py -v
```

3. For $p = 8$ pipeline stages, how many micro-batches $m$ do you need to push
   the 1F1B bubble fraction below $10\%$? How many micro-batches of activations
   does Stage 0 hold at that $m$ under 1F1B vs GPipe?

---

## What you should see

Adding Sequence Parallelism to Tensor Parallelism **divides the entire layer's
activation footprint by $t$** ($34/t$ instead of $10 + 24/t$) with zero
additional network bytes. When you also pipeline layers over multiple nodes, **1F1B
decouples bubble reduction (large $m$) from activation memory (bounded by $p$)**.

---

## Go deeper

- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) (Korthikanti et al., 2022), Section 4.2.2 derives Sequence Parallelism's $g$ and $\bar{g}$ conjugate operators.
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) (Qi et al., 2024), splitting $B_x$ and $B_w$ to fill pipeline bubbles.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), Section 3.2.1 introduces DualPipe.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (HuggingFace Nanotron, 2025), visual walkthrough of 5D parallelism.

---

## Check yourself

1. Why does replacing TP's `all-reduce` with SP's `reduce-scatter` + `all-gather`
   cost zero extra network bytes?
2. Why does Stage 0 hold $p$ micro-batches of activations under 1F1B while Stage
   $p-1$ holds only $1$?
3. Why can $B_w$ ($\nabla_W = X^\top \nabla_Y$) be delayed to fill a pipeline
   bubble while $B_x$ ($\nabla_X = \nabla_Y W^\top$) cannot?

---

## Next

**[34b. Context parallelism, expert parallelism, and the 5D playbook](34b-context-and-expert-parallelism.md)**:
how Zig-Zag Ring Attention balances causal long-context FLOPs over GPUs and how Expert Parallelism shards MoEs over nodes.

```bash
uv run python book/code/context_parallel_balance.py
```

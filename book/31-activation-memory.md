# 31. Activations, checkpointing, and gradient accumulation

**Demo:** `book/code/activation_budget.py` · **Build:** `train/checkpointing.py` · **Test:** `tests/test_31_activations.py`
**Moves:** activation VRAM down $7.7\times$ (FlashAttention) to $147\times$ (full checkpointing)
**Prereq:** [30. The training ledger](30-training-arithmetic.md), [17. FlashAttention](17-flash-attention.md), [14e. Activation lifetimes](14e-activation-lifetimes.md)

---

## The problem

Lecture 30 counted **8.88 GiB** of persistent state for Qwen3-0.6B. On a 24 GiB
RTX 3090, that leaves 15.12 GiB free, which looks like plenty of room for a
0.6B model.

Then we run a training step at `batch_size = 4, seq_len = 4096` and PyTorch
crashes right away with `CUDA out of memory`.

Why does that happen? In inference (Lecture 14e), a layer's intermediate values
die the instant the next layer reads them, so three buffers can share one slot
in memory. In training, Lecture 30's weight-gradient rule

$$\nabla_W = X^\top \nabla_Y$$

forces **every layer** $\ell \in \{1, \dots, L\}$ to keep its forward input
$X_\ell$ in VRAM until the backward pass walks all the way back from layer $L$
to layer $\ell$. Worse still, at the very top of the network sits a single
output tensor of shape `[batch, seq_len, vocab_size]` that quietly takes
**9.27 GiB** all by itself.

---

## See it

```bash
uv run python book/code/activation_budget.py
```

```
Qwen3-0.6B activation memory at batch=4, seq_len=4096 (28 layers)
=========================================================================
  model states (16*Psi, fixed)              8.88 GiB
  activations, eager attention            128.62 GiB  (S and P are O(s^2))
  activations, FlashAttention              16.65 GiB  ( 7.7x smaller)
  activations, full checkpointing          896.0 MiB  (19.0x smaller than Flash)
  unfused fp32 output logits [b, s, V]      9.27 GiB  <-- single-tensor trap!

Total VRAM (states + activations, excluding unfused logits) on 24 GiB 3090
=========================================================================
  batch   seq_len    eager attn    flash attn    flash + ckpt   fits 24 GiB?
      1      2048      17.96 GiB     10.96 GiB        8.99 GiB   all 3
      2      4096      73.19 GiB     17.21 GiB        9.32 GiB   flash/ckpt
      4      4096     137.51 GiB     25.53 GiB        9.76 GiB   ckpt only
      8      4096     266.13 GiB     42.19 GiB       10.63 GiB   ckpt only
```

At `batch=4, seq_len=4096`, eager attention needs **128.62 GiB** of saved
forward values on a 0.6B model. Even with FlashAttention (Lecture 17) it needs
**16.65 GiB**, pushing total memory to $8.88 + 16.65 = 25.53\text{ GiB}$, right
past the 24 GiB limit. Full checkpointing drops that 16.65 GiB to **896.0 MiB**.

---

## The idea

### 1. The lifetime pyramid: why inference aliases and training hoards

Compare the lifetime of layer $\ell$'s activations in inference vs training:

```
INFERENCE (Lecture 14e: constant memory, O(1) in depth L)
  layer 1 writes X_1 -> layer 2 reads X_1, overwrites buffer with X_2 -> ...
  live at any moment: 1 layer's buffers

TRAINING (O(L) in depth: first-in, last-out stack)
  fwd 1 (save X_1) -> fwd 2 (save X_2) -> ... -> fwd 28 (save X_28)
                                                      |
  bwd 1 (free X_1) <- bwd 2 (free X_2) <- ... <- bwd 28 (use & free X_28)
  live at the peak (between fwd 28 and bwd 28): ALL 28 layers' buffers
```

### 2. Deriving per-layer activation memory for Qwen3-0.6B

Following Korthikanti et al. (2022) and adapting their count to Qwen3-0.6B
(GQA, SwiGLU, and RMSNorm without dropout), we can tally every tensor saved
during the forward pass of **one** layer in BF16 (2 bytes per element).

For Qwen3-0.6B, we have $L = 28$ layers and hidden width $h = 1024$. Each layer
has $n_h = 16$ query heads and $n_{kv} = 8$ KV heads with head width
$d_h = 128$, so $n_h d_h = 2048 = 2h$ and $n_{kv} d_h = 1024 = h$. The SwiGLU
MLP expands to intermediate width $d_{\text{ff}} = 3072 = 3h$. Let $b$ be the
batch size and $s$ be the sequence length.

Inspect one transformer layer from input to output and count what backward
must keep in memory.

First, each block runs two RMSNorms, one before attention and one before the
MLP. Differentiating $y = x \cdot w / \sqrt{\text{mean}(x^2) + \epsilon}$ needs
the input vector $x$ at both norms, which takes $2 \times (2 s b h) = 4 s b h$
bytes.

Second, the query, key, and value projections $Q = X W_Q$, $K = X W_K$, and
$V = X W_V$ all read the same post-norm input $X$. To compute the weight
gradients $\nabla_{W_Q}$, $\nabla_{W_K}$, and $\nabla_{W_V}$, we save that shared
input $X$ once, adding $2 s b h$ bytes.

Third, the attention backward pass needs the $Q$, $K$, and $V$ tensors
themselves to compute $\nabla_Q$, $\nabla_K$, and $\nabla_V$. Here $Q$ has shape
$(b, s, 2048)$ for $4 s b h$ bytes, while $K$ and $V$ each have shape
$(b, s, 1024)$ for $4 s b h$ bytes together.

$$\text{QKV outputs} = 2 s b (n_h d_h + 2 n_{kv} d_h) = 8 s b h \text{ bytes}$$

Fourth, the attention output projection $O = \text{Attn}(Q, K, V) W_O$ needs its
input of shape $(b, s, 2048)$ saved for $\nabla_{W_O}$, which takes
$2 s b (n_h d_h) = 4 s b h$ bytes.

Fifth, the MLP gate and up projections $\text{gate} = X W_{\text{gate}}$ and
$\text{up} = X W_{\text{up}}$ share the second post-norm input $X$, adding
another $2 s b h$ bytes.

Sixth, the SwiGLU layer computes $\text{SiLU}(g) \odot u$ and multiplies that
gated result by $W_{\text{down}}$. Differentiating the elementwise product
needs both $g$ and $u$, each of shape $(b, s, d_{\text{ff}})$, and differentiating
$W_{\text{down}}$ needs the gated product itself. Saving all three
tensors of width $d_{\text{ff}} = 3h$ takes:

$$3 \times (2 s b d_{\text{ff}}) = 6 s b (3h) = 18 s b h \text{ bytes}$$

Sum all six linear, norm, and MLP terms:

$$\text{Non-score activations per layer} = (4 + 2 + 8 + 4 + 2 + 18) s b h = 38 s b h \text{ bytes}$$

Now we add the seventh piece, the internal state of attention itself. Its size
depends on whether we run eager attention or FlashAttention (Lecture 17). Eager
attention writes both the pre-softmax scores
$S \in \mathbb{R}^{b \times n_h \times s \times s}$ and the softmax weights
$P \in \mathbb{R}^{b \times n_h \times s \times s}$ to HBM in BF16:

$$\text{Attn}_{\text{eager}} = 2 \times (2 b n_h s^2) = 4 b n_h s^2 \text{ bytes}$$

FlashAttention never writes $S$ or $P$ to HBM (Lecture 17). For backward, it
saves only the FP32 log-sum-exp vector $L \in \mathbb{R}^{b \times n_h \times s}$
($4 b n_h s$ bytes) and rebuilds the $S$ and $P$ tiles on-chip inside SRAM
during the backward pass.

Plug in $b = 4, s = 4096, h = 1024, n_h = 16, L = 28$ term by term:

```
1. Non-score activations (all 28 layers):
   per layer:   38 × s × b × h  =  38 × 4096 × 4 × 1024  =   637,534,208 B  (608.0 MiB)
   28 layers:   28 × 608.0 MiB                           =  17,024.0 MiB    ( 16.63 GiB)

2a. Eager attention score + prob matrices (all 28 layers):
   per layer:   4 × b × n_h × s² =  4 × 4 × 16 × 4096²   = 4,294,967,296 B  (  4.00 GiB)
   28 layers:   28 × 4.00 GiB                            =                   112.00 GiB
   eager total: 16.63 GiB + 112.00 GiB                   =                   128.62 GiB

2b. FlashAttention LSE state (all 28 layers):
   per layer:   4 × b × n_h × s  =  4 × 4 × 16 × 4096    =     1,048,576 B  (  1.00 MiB)
   28 layers:   28 × 1.00 MiB                            =        28.0 MiB  (  0.03 GiB)
   Flash total: 16.63 GiB + 0.03 GiB                     =                    16.65 GiB
```

FlashAttention cuts activation memory from **128.62 GiB to 16.65 GiB** ($7.7\times$)
by eliminating the $112\text{ GiB}$ quadratic attention state.

??? question "Wait, if FlashAttention doesn't save the $s \times s$ attention matrix $P$, how can it compute the backward pass?"
    The standard backward pass uses $P$ to compute $\nabla_V = P^\top \nabla_O$
    and $\nabla_S = P \odot (\nabla_P - \text{rowsum}(\nabla_P \odot P))$.
    FlashAttention's backward kernel reloads the $Q_i, K_j$ tiles into SRAM,
    recomputes $S_{ij} = Q_i K_j^\top / \sqrt{d}$, and rebuilds $P_{ij} = \exp(S_{ij} - L_i)$
    on the fly using the saved 1-D log-sum-exp vector $L_i$ ($1\text{ MiB}$ per
    layer instead of $4\text{ GiB}$). It trades a small amount of extra tensor-core
    math to avoid reading and writing $112\text{ GiB}$ of HBM traffic, so the
    backward pass finishes *faster* despite recomputing $P$.
    [Full answer](qa.md#if-flashattention-doesnt-save-the-sxs-matrix-p-how-does-its-backward-pass-work)

---

### 3. Activation checkpointing (gradient checkpointing)

Even with FlashAttention, $16.65\text{ GiB}$ of saved tensors plus $8.88\text{ GiB}$
of model state comes to $25.53\text{ GiB}$, which still overflows a 24 GiB card.

We can apply the same trade FlashAttention made to the whole transformer block.
Instead of saving all the internal tensors inside each layer, we save only the
layer-boundary input $X_\ell$, and we rerun the layer's forward pass on the fly
right before its backward pass runs.

With **full activation checkpointing**, each of the $L = 28$ layers saves only
its boundary input tensor of shape $(b, s, h)$ in BF16.

$$\text{Bytes}_{\text{ckpt}} = L \times (2 s b h)$$

Plug in $L = 28, b = 4, s = 4096, h = 1024$:

```
per layer boundary:  2 × 4096 × 4 × 1024  =   33,554,432 B  =   32.0 MiB
all 28 layers:       28 × 32.0 MiB        =  896,000,000 B  =  896.0 MiB  (0.875 GiB)

reduction vs FlashAttention:  16.65 GiB / 0.875 GiB  =  19.0×
reduction vs eager attention: 128.62 GiB / 0.875 GiB = 147.0×
```

What do we pay for that $19\times$ memory drop? Each layer runs its forward pass
**twice**, once on the way up and once right before its backward pass, so total
compute rises from $6ND$ to $8ND$.

$$\text{FLOPs}_{\text{ckpt}} = \underbrace{2ND}_{\text{forward}} + \underbrace{2ND}_{\text{recompute forward}} + \underbrace{4ND}_{\text{backward}} = 8ND \quad (+33\% \text{ FLOPs})$$

**Selective checkpointing** (Korthikanti et al.) notices that tensors inside a
layer have very different bytes-per-FLOP ratios. SwiGLU intermediates ($18 s b h$)
and RMSNorms take almost no FLOPs to recompute, while the large linear matmuls
account for nearly all the $2ND$ FLOPs. Recomputing only the cheap, fat ops
(norms, SwiGLU gate product, and attention) saves most of the memory for under
$5\%$ extra compute.

---

### 4. The output-head logit trap: 9.27 GiB in a single tensor

Look at the last line of the demo's first table:

```
unfused fp32 output logits [b, s, V]      9.27 GiB  <-- single-tensor trap!
```

Qwen3 has a vocabulary of $V = 151{,}936$ tokens. At the top of layer 28, the
output head computes logits of shape $(b, s, V)$, and a standard call to
`F.cross_entropy(logits.float(), labels)` casts that whole tensor to FP32:

```
elements  =  b × s × V         =  4 × 4096 × 151,936  =  2,489,319,424 floats
fp32 size =  2,489,319,424 × 4 =  9,957,277,696 B     =  9.27 GiB (and +9.27 GiB for grad!)
```

All 28 checkpointed layers took only **0.88 GiB** together, yet one naive
`logits.float()` call allocates **9.27 GiB for forward plus 9.27 GiB for its
gradient**, crashing a 24 GiB GPU on the spot.

??? question "How do you compute cross-entropy over 151,936 vocabulary tokens without materializing the 9.27 GiB logit tensor?"
    Two ways, both standard in modern trainers (Liger-Kernel, Unsloth, Megatron):
    **chunked cross-entropy** splits the $b \cdot s = 16{,}384$ token rows into
    chunks of e.g. $1{,}024$ tokens, computing the $1{,}024 \times 151{,}936$
    matmul + loss + backward one chunk at a time ($9.27 / 16 = 0.58\text{ GiB}$
    peak); **fused linear cross-entropy** goes further and computes the online
    log-sum-exp across vocabulary tiles inside a Triton kernel (the exact online
    softmax from Lecture 17!) so the full logit tensor is never written to HBM at all.
    [Full answer](qa.md#how-do-you-compute-cross-entropy-without-materializing-the-927-gib-logit-tensor)

---

### 5. Gradient accumulation: decoupling convergence batch from VRAM batch

Optimization theory wants a large **global batch size** (often $B_{\text{global}} = 1\text{M}$
to $4\text{M}$ tokens per optimizer step so gradient noise is low). Your GPU
VRAM only fits a tiny **micro-batch** of $b = 2$ or $b = 4$ sequences ($8{,}192$
to $16{,}384$ tokens).

**Gradient accumulation** bridges the two without buying more GPUs: run $K$
forward+backward micro-steps sequentially, **adding** each micro-step's
gradients into the same $g_{\text{bf16}}$ buffer, and call `optimizer.step()`
only on step $K$:

```python
optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    loss = forward_and_loss(micro_batch[micro_step]) / grad_accum_steps
    loss.backward()          # frees this micro-batch's activations immediately!
optimizer.step()             # updates weights using the average over K micro-batches
```

Why activation memory stays at $1\times$ a single micro-batch: `loss.backward()`
consumes and frees micro-batch $k$'s activations *before* micro-batch $k+1$'s
forward pass allocates its own. You trade wall-clock time ($K$ sequential steps)
for an arbitrarily large effective batch size at fixed VRAM.

---

## The code

```python
def layer_activation_bytes(
    seq_len: int,
    batch: int,
    hidden: int = 1024,
    n_heads: int = 16,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    intermediate: int = 3072,
    flash: bool = True,
) -> int:
    s, b, h = seq_len, batch, hidden
    linear_and_norm = (
        4 * s * b * h
        + 2 * s * b * h
        + 2 * s * b * (n_heads * head_dim + 2 * n_kv_heads * head_dim)
        + 2 * s * b * (n_heads * head_dim)
        + 2 * s * b * h
        + 6 * s * b * intermediate
    )
    attn = (4 * b * n_heads * s) if flash else (4 * b * n_heads * s * s)
    return linear_and_norm + attn


def checkpointed_activation_bytes(
    seq_len: int, batch: int, n_layers: int = 28, hidden: int = 1024
) -> int:
    return n_layers * 2 * seq_len * batch * hidden


def logit_bytes(
    seq_len: int, batch: int, vocab_size: int = 151_936, dtype_bytes: int = 4
) -> int:
    return seq_len * batch * vocab_size * dtype_bytes
```

---

## Build it

1. Implement `layer_activation_bytes`, `checkpointed_activation_bytes`, and
   `logit_bytes` in `train/checkpointing.py`.
2. Run the tests:

```bash
uv run pytest tests/test_31_activations.py -v
```

3. Calculate how many `grad_accum_steps` you need on a single RTX 3090 (using
   micro-batch $b = 4, s = 4096$ with FlashAttention + activation checkpointing)
   to reach a 1-million-token global batch size.

---

## What you should see

Eager attention grows as $O(s^2)$ in memory and hits **128.62 GiB** at
$b=4, s=4096$, while FlashAttention brings that down to $O(s)$ at **16.65 GiB**.
Turning on full checkpointing drops all 28 layers of saved tensors to **896 MiB**,
so even $b=8, s=4096$ fits on a 24 GiB 3090 as long as we chunk or fuse the
output-head cross-entropy.

---

## Go deeper

- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) (Korthikanti et al., 2022), the per-layer memory count and selective checkpointing.
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (Chen et al., 2016), the original gradient-checkpointing paper.
- [Liger-Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/abs/2410.10989), fused linear cross-entropy, RMSNorm, and SwiGLU kernels that remove the 9.27 GiB logit spike.

---

## Check yourself

1. Why does $\nabla_W = X^\top \nabla_Y$ require storing forward activations across
   all $L$ layers simultaneously, whereas inference only needs one layer's
   buffers at a time?
2. What single vector does FlashAttention save per layer for the backward pass,
   and why is that enough to reconstruct $P$?
3. Why does PyTorch's `model.gradient_checkpointing_enable()` still OOM if you
   call `F.cross_entropy(logits.float(), labels)` on a 151,936-vocab model at
   $b=4, s=4096$?
4. Why does `loss.backward()` inside the gradient-accumulation loop keep peak
   activation memory at $1\times$ a micro-batch instead of $K\times$?

---

## Next

**[32. Data pipelines, sequence packing, and loss masking](32-data-and-packing.md)**:
your batch now fits in VRAM, yet 58% of its FLOPs are computing on `[PAD]` tokens.

```bash
uv run python book/code/packing_waste.py
```

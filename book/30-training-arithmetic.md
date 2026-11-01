# 30. The training ledger: $6ND$ and the $16\Psi$ wall

**Demo:** `book/code/training_ledger.py` · **Build:** `train/ledger.py` · **Test:** `tests/test_30_training_ledger.py`
**Moves:** predicts training VRAM and step time before a single backward pass runs
**Prereq:** [02. Arithmetic intensity](02-arithmetic-intensity.md), [22. Tensor parallelism](22-tensor-parallelism.md)

---

## The problem

Parts I through V treated the weights as a read-only artifact: 840 MiB of
16-bit numbers (1.11 GiB including the tied token embedding) sitting in VRAM,
streamed onto the chip once per prefill or once per decoded token.

The moment you call `loss.backward()` and `optimizer.step()`, that mental model
breaks in two places at once. Memory multiplies by $8\times$ before a single
token runs: a model whose weights take 1.11 GiB in BF16 for inference needs
**8.88 GiB** of resident model state under mixed-precision AdamW, even before
counting the forward activations saved for the backward pass (Lecture 31). At
the same time, compute per token triples. A forward pass over $D$ tokens with
$N$ non-embedding parameters costs $2ND$ FLOPs (Lecture 01), and the backward
pass adds $4ND$ FLOPs, bringing total training compute to **$6ND$ FLOPs per
token**.

Every systems technique in Part VI (activation checkpointing, ZeRO, FSDP,
sequence parallelism, pipeline schedules, FP8 training) attacks a specific row
of this ledger. Before building any of them, derive the ledger from scratch.

---

## See it

```bash
uv run python book/code/training_ledger.py
```

Pure arithmetic, no GPU required:

```
Qwen3-0.6B model-state ledger (596.0M total params, incl. tied embedding)
=========================================================================
  inference weights (bf16, 2 B/param)       1136.9 MiB  (1.11 GiB)
  + gradients       (bf16, 2 B/param)       1136.9 MiB
  + master weights  (fp32, 4 B/param)       2273.8 MiB
  + Adam first mom  (fp32, 4 B/param)     2273.8 MiB
  + Adam second mom (fp32, 4 B/param)     2273.8 MiB
  -----------------------------------------------------------------------
  training state    (16 B/param)            9095.0 MiB  (8.88 GiB)  -> 8.0x inference

Compute per 4096-token sequence (440.4M non-embedding params)
=========================================================================
  forward           (Y = X @ W)             3607.8 GFLOP  (2ND)
  backward input    (dX = dY @ W^T)         3607.8 GFLOP  (2ND)
  backward weight   (dW = X^T @ dY)         3607.8 GFLOP  (2ND)
  total training    (6ND)                  10823.3 GFLOP  (3.0x prefill)

Why FP32 master weights exist (w=1.0, 1000 steps of delta=-1e-4)
=========================================================================
  FP32 master weight after 1000 steps : 0.9000  (moved by -0.1000)
  BF16 weight without master copy     : 1.0000  (stuck! ULP at 1.0 is 1/128 = 0.0078125)
```

Look at the three blocks in order: **8.88 GiB** of persistent state for a
"0.6B" model, **10,823.3 GFLOP** per 4096-token sequence, and a BF16 weight
that receives 1,000 gradient updates and **never moves at all**.

---

## The idea

### 1. Why training is $6ND$ FLOPs, derived on one matrix multiply

Take a single linear layer without bias. Let the input activations be
$X \in \mathbb{R}^{D \times d_{\text{in}}}$, where $D = B \cdot S$ is the number
of tokens in the batch, and let the weight matrix be
$W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$. The layer has
$P = d_{\text{in}} d_{\text{out}}$ parameters.

**Forward pass.** The layer computes:

$$Y = X W, \qquad Y \in \mathbb{R}^{D \times d_{\text{out}}}$$

Each of the $D \cdot d_{\text{out}}$ output entries is a dot product of length
$d_{\text{in}}$, which is $d_{\text{in}}$ multiplies and $d_{\text{in}}$ adds
($2 d_{\text{in}}$ FLOPs):

$$\text{FLOPs}_{\text{fwd}} = D \cdot d_{\text{out}} \cdot (2 d_{\text{in}}) = 2 D (d_{\text{in}} d_{\text{out}}) = 2 P D$$

**Backward pass.** Backpropagation hands this layer the upstream gradient
$\nabla_Y = \frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{D \times d_{\text{out}}}$.
The layer must produce **two** gradients for two different consumers.

First, it computes the backward gradient for the inputs ($B_x$), which is the
gradient with respect to $X$ that passes to the previous layer:

$$\nabla_X = \nabla_Y W^\top, \qquad (D \times d_{\text{out}}) \times (d_{\text{out}} \times d_{\text{in}}) \to (D \times d_{\text{in}})$$

Count the operations. There are $D \cdot d_{\text{in}}$ output entries, and each
is a dot product of length $d_{\text{out}}$ ($2 d_{\text{out}}$ FLOPs):

$$\text{FLOPs}_{B_x} = D \cdot d_{\text{in}} \cdot (2 d_{\text{out}}) = 2 P D$$

Second, it computes the backward gradient for the weights ($B_w$), which is the
gradient with respect to $W$ handed to the optimizer so this layer's weights can
update:

$$\nabla_W = X^\top \nabla_Y, \qquad (d_{\text{in}} \times D) \times (D \times d_{\text{out}}) \to (d_{\text{in}} \times d_{\text{out}})$$

Count the operations again. Each of the $d_{\text{in}} \cdot d_{\text{out}}$
output entries is a dot product over all $D$ tokens ($2 D$ FLOPs):

$$\text{FLOPs}_{B_w} = d_{\text{in}} \cdot d_{\text{out}} \cdot (2 D) = 2 P D$$

Add the three matrix multiplies together:

$$\text{FLOPs}_{\text{train}} = \underbrace{2PD}_{\text{forward } (Y = XW)} + \underbrace{2PD}_{\text{input grad } (\nabla_X = \nabla_Y W^\top)} + \underbrace{2PD}_{\text{weight grad } (\nabla_W = X^\top \nabla_Y)} = 6 P D$$

Summed across all non-embedding parameters $N = \sum_{\ell} P_\ell$ in the
network ($N = 440{,}401{,}920$ for Qwen3-0.6B, from Lecture 01), a training
pass over $D = 4096$ tokens does:

```
forward  (2ND):  2 × 440,401,920 × 4096  =   3,607,772,528,640 FLOP  =   3,607.8 GFLOP
backward (4ND):  4 × 440,401,920 × 4096  =   7,215,545,057,280 FLOP  =   7,215.5 GFLOP
total    (6ND):  6 × 440,401,920 × 4096  =  10,823,317,585,920 FLOP  =  10,823.3 GFLOP
```

Notice the structure of the two backward matmuls, because three later lectures
depend on it directly:

- $\nabla_W = X^\top \nabla_Y$ requires the forward input **$X$** to still be in
  memory when the backward pass arrives at this layer. That single requirement
  is why training has an activation-memory problem (Lecture 31) and inference
  does not.
- $\nabla_X = \nabla_Y W^\top$ is on the **critical path** to the previous
  layer, while $\nabla_W = X^\top \nabla_Y$ is needed only locally before
  `optimizer.step()`. Splitting $B_x$ from $B_w$ is what makes Zero-Bubble
  pipeline parallelism and DeepSeek-V3's DualPipe work (Lecture 34).
- The three matmuls contract along **three different axes**: $d_{\text{in}}$ in
  forward, $d_{\text{out}}$ in $B_x$, and the token axis $D$ in $B_w$. That is
  why FP8 training needs transposed / block-wise quantization layouts that FP8
  inference never had to build (Lecture 35).

??? question "Why does the $6ND$ formula use the 440.4M non-embedding parameters, while the memory ledger uses 596.0M?"
    Because the token embedding table ($151{,}936 \times 1024 = 155.6\text{M}$
    parameters) is a **lookup table** on the input side, which costs zero
    multiply-adds, and on the output side Qwen3 **ties** the language-modeling
    head to that same embedding matrix. In the $6ND$ convention (Kaplan et al.,
    Hoffmann et al.), $N$ counts non-embedding parameters; if you explicitly
    include the output-head logit projection, $N$ rises to $596.0\text{M}$. By
    contrast, **VRAM must hold every parameter tensor**, and the optimizer
    maintains state for the embedding table too — so the memory ledger always
    uses the full $\Psi = 596{,}049{,}920$ parameters.
    [Full answer](qa.md#why-does-the-6nd-formula-use-4404m-parameters-while-the-memory-ledger-uses-5960m)

---

### 2. Why mixed-precision AdamW costs $16\Psi$ bytes

Let $\Psi$ be the total number of trainable parameters in the model
($\Psi = 596{,}049{,}920$ for Qwen3-0.6B). Why can't we keep weights,
gradients, and optimizer states in BF16 (2 bytes each)? Weight updates are
thousands of times smaller than the weights themselves, so adding a tiny step
directly to a BF16 weight rounds the update away to zero.

Recall the bit layout of 16-bit and 32-bit floats:

```
Format   Sign   Exponent   Mantissa (fraction bits)   Unit in Last Place (ULP) near 1.0
---------------------------------------------------------------------------------------
FP32      1b       8b                23b              2^-23  ≈  1.19 × 10^-7
BF16      1b       8b                 7b              2^-7   =  1 / 128 = 0.0078125
FP16      1b       5b                10b              2^-10  =  1 / 1024 ≈ 0.0009766
```

Suppose a weight sits at $w = 1.0$. In BF16, the next representable number
below $1.0$ is $1.0 - 2^{-8} = 0.99609375$, and the spacing above $1.0$ is
$2^{-7} = 0.0078125$. Under nearest-even rounding, any perturbation $|\Delta w|$
smaller than half a step ($2^{-8} \approx 3.9 \times 10^{-3}$ above $1.0$, or
$2^{-9} \approx 1.95 \times 10^{-3}$ below $1.0$) rounds straight back to $1.0$:

$$\text{round}_{\text{bf16}}(1.0 - 10^{-4}) = 1.0$$

In real LLM training, learning rates sit around $\eta \in [10^{-5}, 3 \times 10^{-4}]$.
Even when AdamW normalizes the gradient so $m_t / \sqrt{v_t} \approx 1$, the
per-step update is $|\Delta w| \approx \eta \sim 10^{-4}$, which is **$20\times$
to $40\times$ smaller than BF16's rounding threshold** near $1.0$. Without a
higher-precision accumulator, the weight never moves.

??? question "Why use BF16 for the matmuls if we still have to keep FP32 master weights?"
    Because the optimizer update $w \leftarrow w - \eta \cdot m / \sqrt{v}$ is a
    cheap elementwise pass run **once per step**, whereas the matrix multiplies
    ($6ND$ FLOPs) dominate compute and activation memory. Tensor cores run BF16
    matmuls at $2\times$ to $8\times$ the rate of FP32 and halve the size of every
    saved activation in Lecture 31. So mixed-precision training uses BF16 for
    the forward/backward matmuls and casts to FP32 only inside the optimizer step.
    [Full answer](qa.md#why-use-bf16-for-matmuls-if-we-still-keep-fp32-master-weights)

Count the bytes per parameter that mixed-precision AdamW must hold in VRAM.
Forward and backward matrix multiplies read the working weights $w_{\text{bf16}}$
in BF16 (**2 bytes/param**), and the backward pass writes gradients
$g_{\text{bf16}}$ in BF16 (**2 bytes/param**; if you accumulate gradients in
FP32 across many micro-batches, this copy takes 4 bytes/param and raises the
total to $18\Psi$). Alongside those 16-bit tensors, the optimizer keeps FP32
master weights $w_{\text{fp32}}$ where small updates accumulate without
underflowing (**4 bytes/param**), the FP32 first moment
$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ (**4 bytes/param**), and the FP32
second moment $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ (**4 bytes/param**).

Sum the five terms:

$$\text{Bytes}_{\text{state}} = (\underbrace{2}_{w} + \underbrace{2}_{g} + \underbrace{4}_{w_{\text{master}}} + \underbrace{4}_{m} + \underbrace{4}_{v}) \Psi = 16 \Psi \text{ bytes}$$

Plug in Qwen3-0.6B ($\Psi = 596{,}049{,}920$, and $1\text{ MiB} = 1{,}048{,}576\text{ B}$,
$1\text{ GiB} = 1{,}073{,}741{,}824\text{ B}$):

```
w_bf16    =  2 × 596,049,920 B  =  1,192,099,840 B  =  1,136.9 MiB  (1.11 GiB)
g_bf16    =  2 × 596,049,920 B  =  1,192,099,840 B  =  1,136.9 MiB  (1.11 GiB)
w_master  =  4 × 596,049,920 B  =  2,384,199,680 B  =  2,273.8 MiB  (2.22 GiB)
m_fp32    =  4 × 596,049,920 B  =  2,384,199,680 B  =  2,273.8 MiB  (2.22 GiB)
v_fp32    =  4 × 596,049,920 B  =  2,384,199,680 B  =  2,273.8 MiB  (2.22 GiB)
------------------------------------------------------------------------------
total     = 16 × 596,049,920 B  =  9,536,798,720 B  =  9,095.0 MiB  (8.88 GiB)
```

And note what fraction of that 8.88 GiB belongs to the optimizer:
$(4 + 4 + 4) / 16 = 12 / 16 = 75\%$. Three quarters of your model-state VRAM
holds AdamW's bookkeeping, while the working weights and gradients take only one
quarter. That $75\%$ share is the target of ZeRO-1 in Lecture 33.

---

### 3. MFU vs HFU: the ruler for training speed

In inference, the ruler was TTFT and TPOT (Lecture 04). In training, nobody is
waiting on one token; the ruler is **Model FLOPs Utilization (MFU)**, introduced
in the PaLM paper:

$$\text{MFU} = \frac{\text{useful model FLOPs per second}}{\text{hardware peak FLOPs per second}} = \frac{6 N \cdot (\text{tokens/sec})}{\text{Peak FLOP/s}}$$

Look closely at the numerator: **$6N$ times throughput**, regardless of how you
implemented the step. If you turn on full activation checkpointing (Lecture 31),
your GPU actually executes an extra forward pass during backward. That runs
$8ND$ FLOPs on the hardware (**Hardware FLOPs Utilization, HFU**), even though
only $6ND$ count as useful progress toward training the model.

Work the distinction on an RTX 3090 ($71\text{ TFLOPS}$ dense BF16/FP16 tensor
core peak) training Qwen3-0.6B ($N = 440.4\text{M}$) at $12{,}000\text{ tokens/sec}$
with full activation checkpointing:

```
useful FLOP/s (6ND)   =  6 × 440.4 × 10^6 × 12,000  =  31.71 TFLOP/s
hardware FLOP/s (8ND) =  8 × 440.4 × 10^6 × 12,000  =  42.28 TFLOP/s

MFU  =  31.71 / 71.0  =  44.7%      (what you report)
HFU  =  42.28 / 71.0  =  59.5%      (what the silicon actually did)
```

Well-tuned distributed LLM runs land between **40% and 60% MFU**. If someone
quotes 75% without saying whether it is MFU or HFU, they are almost certainly
counting recomputation FLOPs in the numerator.

---

### 4. Chinchilla scaling vs inference-aware over-training

How many tokens $D$ should a model of size $N$ train on? Hoffmann et al.
(*Chinchilla*, 2022) minimized final loss under a **fixed training compute
budget** $C = 6ND$ and found that $N$ and $D$ should scale equally:

$$D_{\text{Chinchilla}} \approx 20 N$$

For Qwen3-0.6B ($N \approx 0.6\text{B}$), Chinchilla says to stop after
$D_{\text{Chinchilla}} = 20 \times 0.6 \times 10^9 = 12\text{B tokens}$. Yet
Qwen3-0.6B was actually pre-trained on **36 trillion tokens**
($D = 36{,}000\text{B}$), which is $3{,}000\times$ the Chinchilla token budget:

$$\frac{36{,}000\text{B}}{12\text{B}} = 3{,}000\times \text{ Chinchilla}$$

Teams pay $3{,}000\times$ the compute-optimal training bill because Chinchilla
optimized **training cost alone** and priced inference at zero. From Lecture 01,
every decoded token in production costs $2N$ bytes of weight traffic forever. A
smaller model trained far past the Chinchilla point costs more **once** at
training time to make every future token cheaper at serving time.

---

## The code

```python
def training_flops(params: int, tokens: int, checkpointed: bool = False) -> int:
    mult = 8 if checkpointed else 6
    return mult * params * tokens


def adamw_state_bytes(
    params: int, master_fp32: bool = True, grad_bytes: int = 2
) -> int:
    bytes_per_param = 2 + grad_bytes + 4 + 4 + (4 if master_fp32 else 0)
    return params * bytes_per_param


def mfu(achieved_flops_per_sec: float, peak_flops_per_sec: float) -> float:
    return achieved_flops_per_sec / peak_flops_per_sec
```

---

## Build it

1. Implement `training_flops`, `adamw_state_bytes`, and `mfu` in `train/ledger.py`.
2. Run the gate:

```bash
uv run pytest tests/test_30_training_ledger.py -v
```

3. Compute by hand: what is the largest dense model (in parameters $\Psi$) whose
   $16\Psi$ mixed-precision AdamW state fits inside a 24 GiB RTX 3090 with at
   least 4 GiB left over for activations?

---

## What you should see

Running the ledger gives **8.88 GiB** of persistent state for Qwen3-0.6B, which
is $8\times$ its 1.11 GiB BF16 inference footprint. On a single 24 GiB card,
that sets a ceiling of **~1.34B parameters** before you even allocate
activations ($20\text{ GiB} / 16\text{ B} \approx 1.34\text{B}$). Whenever
activation checkpointing is on, reported **MFU sits strictly below HFU**
because recomputing forward activations adds hardware FLOPs without increasing
the $6ND$ numerator.

---

## Go deeper

- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) (EleutherAI), the standard reference for $6ND$ and $16\Psi$.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (Micikevicius et al., 2018), why FP32 master weights prevent small updates from flushing to zero.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022), the Chinchilla $D \approx 20N$ derivation.

---

## Check yourself

1. Why does the backward pass of a linear layer cost $4PD$ FLOPs while the
   forward pass costs $2PD$? Write out both backward matrix multiplies and what
   each one produces.
2. A weight has value $1.0$ in BF16 and receives an AdamW update of $-2 \times 10^{-4}$
   every step. Without FP32 master weights, what is its value after 10,000 steps?
3. Why does turning on activation checkpointing raise HFU while lowering MFU?
4. Why do modern small models train on $1{,}000\times$ to $3{,}000\times$ more
   tokens than Chinchilla recommends?

---

## Next

**[30b. Scaling laws, learning rate schedules, and mid-training](30b-scaling-laws-and-schedules.md)**:
Derive Chinchilla vs. inference-aware over-training, critical batch size, WSD schedules, and YaRN long-context extension.

```bash
uv run python book/code/scaling_laws.py
```

# 35. Low-precision training (FP8) and modern optimizers

**Demo:** `book/code/fp8_trainingsim.py` · **Build:** `train/fp8_train.py` · **Test:** `tests/test_35_fp8_train.py`
**Moves:** doubles tensor-core GEMM throughput ($2\times$ BF16) without outlier collapse or MoE routing tax
**Prereq:** [19b. FP8](19b-fp8.md), [23. MoE and expert parallelism](23-moe-and-expert-parallelism.md), [30. The training ledger](30-training-arithmetic.md)

---

## The problem

In Lecture 19b, we used 8-bit floating point (**FP8 E4M3**, range $\pm 448$) for
**inference**, where weights are frozen and there is only one matrix multiply
per linear layer ($Y = XW$).

Training in FP8 is much harder for three reasons. First, **outliers in
activations and gradients erase normal values**. E4M3 has only 4 exponent bits
and 3 mantissa bits, with minimum positive subnormal $2^{-9} \approx 0.001953$.
Under per-tensor scaling, a single outlier of $500{,}000$ in a tensor forces the
shared scale factor so high that every ordinary value around $1.0$ divides down
to $0.000896 < 2^{-9}$ and **flushes straight to zero**, wiping out gradients.

Second, **each linear layer has three GEMMs contracting along two different
axes**. Forward ($Y = XW$) and input-backward ($\nabla_X = \nabla_Y W^\top$)
contract along the hidden dimensions $d_{\text{in}}$ and $d_{\text{out}}$, while
weight-backward ($\nabla_W = X^\top \nabla_Y$) contracts along the **token
dimension $D = B \cdot S$**.

Third, **MoE load-balancing losses fight the language-modeling loss**. Standard
MoE training adds an auxiliary penalty $\mathcal{L}_{\text{aux}}$ to prevent
routing collapse (Lecture 23), trading token quality against load balance.

DeepSeek-V3 solved all three on a 671B-parameter model over 14.8 trillion tokens
in **2.788M H800 GPU hours ($5.58M)** with zero irrecoverable loss spikes. Let's
derive the mechanisms behind that run.

---

## See it

```bash
uv run python book/code/fp8_trainingsim.py
```

```
FP8 E4M3 quantization of 255 values of 1.0 + one outlier of 500,000.0
=========================================================================
  per-tensor (1 scale for all 256) : tile-0 value 1.0 -> 0.0000  (erased to zero!)
  1x128 tile (scale per 128 strip) : tile-0 value 1.0 -> 1.0000  (preserved exactly)

Auxiliary-loss-free MoE routing thermostat (4 experts, top-1, gamma=0.05)
=========================================================================
  step 1: routed to expert 0 (gate=0.80, untainted) -> updated biases=[-0.05, 0.05, 0.05, 0.05]
  step 2: routed to expert 1 (gate=0.72, untainted) -> updated biases=[0.0, 0.0, 0.1, 0.1]
  step 3: routed to expert 0 (gate=0.80, untainted) -> updated biases=[-0.05, 0.05, 0.15, 0.15]
  step 4: routed to expert 2 (gate=0.65, untainted) -> updated biases=[0.0, 0.1, 0.1, 0.2]
```

Look at both blocks:
- One outlier in index 255 wipes out all 255 normal values (`1.0 -> 0.0000`) under
  per-tensor FP8 scaling; **$1 \times 128$ tile scaling** quarantines the
  outlier to its own 128-element strip and preserves `1.0000` in the clean tile.
- The **bias thermostat** rotates traffic across experts `0 -> 1 -> 0 -> 2` while
  keeping the gating weight multiplied onto the expert output (`0.80, 0.72, 0.65`)
  completely untainted by the bias!

---

## The idea

### 1. Fine-grained FP8 scaling ($1 \times 128$ tiles and $128 \times 128$ blocks) + FP32 tensor-core promotion

Recall E4M3's representable range from Lecture 19b. Its maximum finite value is
$V_{\max} = 448.0$, its minimum positive normal is $2^{-6} = 0.015625$, and its
minimum positive subnormal is $2^{-9} = \frac{1}{512} \approx 0.001953125$. Any
magnitude below $\frac{1}{2} \times 2^{-9} = 2^{-10} \approx 0.0009766$ rounds
to **exact zero**.

Now trace what happens in the demo when 255 elements equal $1.0$ and element 255
is an activation outlier of $500{,}000.0$. Under **per-tensor scaling**, we
compute one scale for all 256 numbers.

$$s_{\text{tensor}} = \frac{\max_i |x_i|}{448.0} = \frac{500{,}000.0}{448.0} \approx 1{,}116.07$$

Divide the ordinary values $x = 1.0$ by $s_{\text{tensor}}$.

$$\tilde{x} = \frac{1.0}{1{,}116.07} \approx 0.000896 < 2^{-10} \implies \text{round}_{\text{E4M3}}(\tilde{x}) = 0.0$$

Every single ordinary activation flushes to zero.

Under **tile-wise scaling** ($1 \times 128$ tiles for activations and gradients,
$128 \times 128$ blocks for weights), each 128-element strip gets its own FP32
scale factor. In Tile 0 (indices `0..127`, all equal to `1.0`), the scale is
$s_0 = 1.0 / 448.0$, so every `1.0` quantizes to `448.0` and dequantizes back to
exact `1.0000`. In Tile 1 (indices `128..255`, holding the outlier), only the
127 neighbors inside that single tile share the large scale; every other tile in
the tensor stays clean.

And inside the Hopper (H100/H800) tensor core, a second problem appears: the
hardware FP8 `mma` (`wgmma.mma_async`) accumulates partial dot products in a
reduced-precision 14-bit/13-bit internal adder before writing out. Over a
contraction dimension of $K = 4096$ or $K = 7168$, that internal rounding drifts.
DeepSeek-V3 fixes this by accumulating only $N_C = 128$ elements (one tile!) on
the tensor core and **promoting the partial sum to full FP32 in CUDA core
registers every 128 elements**:

```
tile k (128 FP8 × 128 FP8) ──► Tensor Core MMA (128 steps) ──► promote to FP32 register (acc += partial)
```

??? question "Why do activations use $1 \times 128$ tiles while weights use $128 \times 128$ blocks?"
    Because of the three GEMM contraction axes from Lecture 30!
    In forward ($Y = XW$), $W$ is contracted along rows; in input-backward
    ($\nabla_X = \nabla_Y W^\top$), $W^\top$ is contracted along columns. A
    symmetric **$128 \times 128$ block scale** on $W$ works along **both** row
    and column contractions without re-quantizing $W$, whereas activations $X$
    and gradients $\nabla_Y$ are quantized on the fly into $1 \times 128$ strips
    aligned with whichever axis the current GEMM contracts over.
    [Full answer](qa.md#why-do-fp8-activations-use-1x128-tiles-while-weights-use-128x128-blocks)

---

### 2. Auxiliary-loss-free MoE load balancing: the thermostat

In Lecture 23, we saw that Mixture-of-Experts routers collapse onto a few
favorite experts unless regularized. The standard fix adds an auxiliary balance
loss $\alpha \sum_i f_i P_i$ to the cross-entropy objective.

The catch: gradients from $\mathcal{L}_{\text{aux}}$ flow into the router and
the backbone, pushing tokens toward underloaded experts **even when those
experts are wrong for the token**, and distorting the gating weights $g_{i,t}$
that scale each expert's output.

DeepSeek-V3's **auxiliary-loss-free load balancing** separates **who gets picked**
from **how much their answer counts**:

1. Give each routed expert $i \in \{1, \dots, N_r\}$ a non-differentiable scalar
   bias $b_i$ (initialized to $0$).
2. Compute raw token-to-expert affinity $s_{i,t} = \sigma(\mathbf{u}_t^\top \mathbf{e}_i)$.
3. **Selection uses $s_{i,t} + b_i$, but gating weight uses the raw $s_{i,t}$:**
   $$g'_{i,t} = \begin{cases} s_{i,t} & \text{if } s_{i,t} + b_i \in \text{TopK}(\{s_{j,t} + b_j\}, K_r) \\ 0 & \text{otherwise} \end{cases}$$
4. After each training step, count how many tokens $\text{load}_i$ each expert
   received across the global batch, compare to the balanced average
   $\overline{\text{load}} = \frac{1}{N_r}\sum_j \text{load}_j$, and nudge $b_i$
   like a thermostat:
   $$b_i \leftarrow b_i - \gamma \cdot \text{sign}(\text{load}_i - \overline{\text{load}})$$

Overloaded experts have $b_i$ lowered by $\gamma$; starved experts have $b_i$
raised by $\gamma$. No gradient from load balancing ever enters the model's
weights, and the gating multiplier $s_{i,t}$ is never corrupted by $b_i$.

---

### 3. Multi-Token Prediction (MTP) and modern optimizers (AdamW vs Muon)

Two more training-side upgrades pay direct dividends in quality and speed:

**Multi-Token Prediction (MTP).** Standard causal pre-training grades each
position $t$ on predicting only $t + 1$. DeepSeek-V3 attaches $D$ sequential
1-layer MTP modules (sharing the main embedding table and output head) so
position $t$ also predicts $t + 2, \dots, t + D + 1$ along a causal chain. At
training time, this densifies supervision per token read; at inference time, the
MTP head is repurposed directly as a **self-drafting speculative decoder**
(Lecture 12), accepting $\sim 85\%$ of second tokens ($\approx 1.8$ tokens per
step) with no external draft model!

**AdamW vs Muon.** AdamW normalizes each parameter's momentum elementwise by
$\sqrt{v_t} + \epsilon$. For a 2-D hidden weight matrix $W \in \mathbb{R}^{m \times n}$,
elementwise scaling ignores the matrix's singular-value spectrum: a few dominant
singular directions dictate the step while smaller singular directions learn
slowly. **Muon** (Jordan et al., used in Kimi Moonshot and DeepSeek-V4) replaces
Adam's elementwise preconditioner on 2-D hidden matrices with **Newton-Schulz
matrix orthogonalization** of the momentum matrix $M_t$:

$$O_t = \text{NewtonSchulz}_5(M_t) \approx U V^\top \quad \text{where } M_t = U \Sigma V^\top$$

Specifically, after normalizing $X_0 = M_t / \|M_t\|_F$ so all singular values
lie in $(0, 1]$, Muon runs $5$ iterations of the odd quintic matrix polynomial.

$$X_{k+1} = a X_k + b X_k (X_k^\top X_k) + c X_k (X_k^\top X_k)^2, \qquad (a, b, c) = (3.4445, -4.7750, 2.0315)$$

This iteration works without calling `torch.linalg.svd` because when
$X_k = U \Sigma_k V^\top$, the products $X_k (X_k^\top X_k) = U \Sigma_k^3 V^\top$
and $X_k (X_k^\top X_k)^2 = U \Sigma_k^5 V^\top$ share the same singular vectors
$U$ and $V^\top$. Each matrix step therefore applies the scalar polynomial
$\phi(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ to every singular value
$\sigma_i \in (0, 1]$, pushing all $\sigma_i \to 1$ in 5 fast BF16 matrix
multiplications.

Setting all singular values of the update to $1$ forces every orthogonal
direction of the weight matrix to advance at the same rate, while needing only
**one** momentum buffer ($4\Psi$ bytes) instead of AdamW's two ($m_t$ and $v_t$,
$8\Psi$ bytes)! (1-D norms, embeddings, and the LM head remain on AdamW.)

**MuonClip (QK-Clip) for Trillion-Parameter MoE Stability (Kimi K2).** Because
Muon amplifies small singular directions in $W_q$ and $W_k$, attention pre-softmax
logits $S^h = \frac{1}{\sqrt{d_h}} Q^h (K^h)^\top$ can grow past $100$ at $1\text{T}$
scale, causing softmax overflow and loss spikes. **Kimi K2** (`1T` total, `32B`
active, trained on $15.5\text{T}$ tokens with zero loss spikes) introduces
**MuonClip**: after each Muon step, if head $h$'s maximum attention logit
$S_{\max}^h = \max_{i,j} S_{i,j}^h$ exceeds threshold $\tau = 100$, rescale that
head's query and key projection weights directly:

$$\gamma_h = \min\left(1, \frac{\tau}{S_{\max}^h}\right), \qquad W_q^h \leftarrow \gamma_h^{0.5} W_q^h, \qquad W_k^h \leftarrow \gamma_h^{0.5} W_k^h$$

Because $( \gamma_h^{0.5} Q^h ) ( \gamma_h^{0.5} K^h )^\top = \gamma_h Q^h (K^h)^\top$,
this caps the next step's logit at $\tau$ without distorting the attention pattern!

---

## The code

```python
def tile_quantize_fp8(x, tile_size: int = 128, fp8_max: float = 448.0):
    import torch

    orig_shape = x.shape
    flat = x.reshape(-1, tile_size).float()
    amax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / fp8_max
    q = torch.clamp(torch.round(flat / scale), -fp8_max, fp8_max)
    return (q * scale).reshape(orig_shape).to(x.dtype)


def update_expert_biases(
    biases: list[float], loads: list[int], gamma: float
) -> list[float]:
    target = sum(loads) / len(loads)
    out: list[float] = []
    for b, ld in zip(biases, loads):
        if ld > target:
            out.append(b - gamma)
        elif ld < target:
            out.append(b + gamma)
        else:
            out.append(b)
    return out
```

---

## Build it

1. Implement `tile_quantize_fp8` and `update_expert_biases` in `train/fp8_train.py`.
2. Run the tests:

```bash
uv run pytest tests/test_35_fp8_train.py -v
```

3. Verify that in `tile_quantize_fp8`, an outlier in elements `[128..255]` has
   **zero effect** on the reconstruction error of elements `[0..127]`.

---

## What you should see

Tile quantization keeps outlier quantization noise inside the single 128-element
tile that holds the outlier, so every other tile keeps its full precision. In
the MoE router, the bias thermostat converges toward even expert load over
repeated steps without changing the raw gating weights $s_{i,t}$.

---

## Go deeper

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), Section 2.1.2 (Auxiliary-loss-free load balancing), Section 2.2 (Multi-Token Prediction), and Section 3.3 (FP8 mixed-precision training framework).
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) (Wang et al., 2024).
- [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) (Keller Jordan, 2024).

---

## Check yourself

1. Why does a single outlier of $500{,}000$ cause $1.0$ to flush to zero under
   per-tensor FP8 E4M3 quantization? Show the division against $2^{-9}$.
2. Why does DeepSeek-V3 add $b_i$ inside `TopK(s_{i,t} + b_i)` but **not** inside
   the gating weight $g'_{i,t} = s_{i,t}$?
3. How does a Multi-Token Prediction (MTP) head trained during pre-training turn
   into a speculative decoding speedup at inference time?

---

**Part VI complete.** You can now size, shard, pack, pipeline, and quantize a
pre-training run from first principles.

## Next

**[36. Supervised fine-tuning (SFT), LoRA, and QLoRA](36-sft-and-lora.md)**:
Part VII begins. A pre-trained base model only completes documents; now turn it
into an assistant and see why LoRA cuts optimizer memory while leaving
activation memory untouched.

```bash
uv run python book/code/lora_memory.py
```

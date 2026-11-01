# Lecture 34b: Context Parallelism, Expert Parallelism & the 5D Playbook

[Lecture 34](34-3d-parallelism-and-schedules.md) showed how Tensor Parallelism (TP), Sequence Parallelism (SP), and Pipeline Parallelism (PP) shard weights and activations across GPUs. Yet two frontier workloads break 3D parallelism:

1. **Long-context training ($s = 32\text{K}$ to $1\text{M}$ tokens):** Inside every TP attention block, each GPU still processes the **entire sequence length $s$** for its local subset of heads $h/t$, and TP cannot scale past $t = 8$ GPUs (one NVLink node).
2. **Mixture-of-Experts (MoE) training (`Qwen3-235B-A22B`, `DeepSeek-V3`):** Replicating $128$–$256$ experts across data-parallel ranks blows up memory, while sharding tiny expert MLPs with TP incurs excessive intra-layer All-Reduce latency.

This lecture derives the fourth and fifth dimensions of distributed training — **Context Parallelism (CP)** and **Expert Parallelism (EP)** — and synthesizes the complete **5D Parallelism** configuration playbook.

---

### 1. Context Parallelism & Ring Attention

In Sequence Parallelism ([Lecture 34](34-3d-parallelism-and-schedules.md)), tokens are split along the sequence dimension *only* in LayerNorm, Dropout, and residual connections; the All-Gather before `QKV` projection reconstructs the full sequence $s$ on every GPU inside the TP group.

**Context Parallelism (CP)** splits the input sequence of length $s$ into $C$ chunks of length $L_c = s / C$ across $C$ GPUs **everywhere in the network**, including inside attention.

- Every linear projection (`Q, K, V, O`, `gate, up, down`), RMSNorm, and SwiGLU operates independently on each token, so sharding along $s$ requires **zero communication** in MLP and norm layers.
- Only **Self-Attention** requires cross-GPU communication, because query tokens $Q_r$ on GPU $r$ must attend to key-value pairs $(K_j, V_j)$ residing on GPUs $j \neq r$.

**Ring Attention (Liu et al., 2023)** hides this KV communication behind attention compute using FlashAttention's online softmax ([Lecture 17](17-flash-attention.md)):

1.Arrange the $C$ GPUs in a logical ring: $0 \to 1 \to 2 \to \dots \to C-1 \to 0$.
2. In step $0$, GPU $r$ computes local block attention between $Q_r$ and its own $(K_r, V_r)$ while asynchronously sending $(K_r, V_r)$ to GPU $(r+1) \bmod C$ and receiving $(K_{(r-1)\bmod C}, V_{(r-1)\bmod C})$ from GPU $(r-1) \bmod C$.
3. Across $C$ ring steps, GPU $r$ updates its running online-softmax statistics $(m_r, \ell_r, O_r)$ as each KV chunk circulates around the ring.

#### Condition for Hiding Ring Communication Behind Attention Compute

Per ring step on chunk length $L_c = s/C$ with $h_q$ query heads and $h_{kv}$ KV heads of dimension $d_h$ ($d_q = h_q d_h$, $d_{kv} = h_{kv} d_h$):

- **Compute per step (forward, unmasked block):**
  $$\text{FLOPs}_{\text{step}} = 4 b L_c^2 d_q = \frac{4 b s^2 d_q}{C^2}$$
- **P2P KV bytes sent per step (BF16 $K$ and $V$):**
  $$\text{Bytes}_{\text{step}} = 2 \times 2 b L_c d_{kv} = \frac{4 b s d_{kv}}{C}$$
- **Compute-to-communication ratio:**
  $$\frac{\text{FLOPs}_{\text{step}}}{\text{Bytes}_{\text{step}}} = \frac{4 b s^2 d_q / C^2}{4 b s d_{kv} / C} = \frac{s}{C} \cdot \frac{h_q}{h_{kv}} = L_c \cdot g_{\text{GQA}}$$

As long as the per-GPU chunk length $L_c = s/C$ satisfies $L_c \cdot g_{\text{GQA}} > \frac{\text{Peak TFLOP/s}}{\text{Link GB/s}}$, P2P KV transfers are completely hidden under attention compute!

---

### 2. Why Naive Causal Ring Attention Wastes Nearly Half Your GPUs — and How Zig-Zag Fixes It

Suppose we split a sequence of length $s$ contiguously across $C$ GPUs so GPU $r \in \{0, 1, \dots, C-1\}$ holds token indices $[r L_c,\; (r+1) L_c)$.

Under a **causal mask** ($k \le q$), how many valid $(q, k)$ pairs must GPU $r$ evaluate across the full ring?

#### Full Derivation of Naive vs. Zig-Zag Load Balance

1. **Naive contiguous assignment:**
   - Queries on GPU $r$ attend **fully** to all $r$ earlier chunks $j \in \{0, \dots, r-1\}$, each contributing $L_c^2$ token pairs ($r L_c^2$ total).
   - Queries on GPU $r$ attend **causally** (lower triangle) to their own chunk $j = r$, contributing $\frac{1}{2} L_c^2$ token pairs.
   - Queries on GPU $r$ attend to **zero** tokens in future chunks $j > r$.
   - Total causal work on GPU $r$:
     $$\text{Work}_{\text{naive}}(r) = \left(r + \frac{1}{2}\right) L_c^2 = (2r + 1) \cdot \frac{1}{2} L_c^2$$

   Compare GPU $0$ (first chunk) against GPU $C-1$ (last chunk):

   $$\frac{\text{Work}_{\text{naive}}(C-1)}{\text{Work}_{\text{naive}}(0)} = \frac{2(C-1) + 1}{2(0) + 1} = 2C - 1$$

   And the cluster attention utilization (average work divided by the straggler GPU $C-1$) is:

   $$\text{Util}_{\text{naive}}(C) = \frac{\frac{1}{C}\sum_{r=0}^{C-1} (2r+1)}{2C - 1} = \frac{C^2 / C}{2C - 1} = \frac{C}{2C - 1}$$

   At $C = 4$, GPU $3$ does **$7\times$** the work of GPU $0$ ($\text{Util} = 4/7 = 57.1\%$). At $C = 8$, GPU $7$ does **$15\times$** the work of GPU $0$ ($\text{Util} = 8/15 = 53.3\%$)!

2. **Zig-Zag Ring Attention (Striped / Symmetric Pairing):**
   Instead of $C$ contiguous chunks of length $L_c = s/C$, divide the sequence into **$2C$ sub-chunks** of length $L_z = \frac{s}{2C} = \frac{1}{2} L_c$, indexed $0, 1, \dots, 2C-1$. Assign GPU $r$ the **symmetric pair** of sub-chunks:

   $$\text{Sub-chunks}(r) = \{\, r, \;\; 2C - 1 - r \,\}$$

   For $C = 4$ ($8$ sub-chunks $0 \dots 7$):
   - GPU $0$ gets $\{0, 7\}$
   - GPU $1$ gets $\{1, 6\}$
   - GPU $2$ gets $\{2, 5\}$
   - GPU $3$ gets $\{3, 4\}$

   Now sum the causal work of both sub-chunks on GPU $r$:
   - Sub-chunk $a = r$ evaluates $\left(r + \frac{1}{2}\right) L_z^2$ causal pairs.
   - Sub-chunk $b = 2C - 1 - r$ evaluates $\left(2C - 1 - r + \frac{1}{2}\right) L_z^2$ causal pairs.
   - Total work on GPU $r$:
     $$\text{Work}_{\text{zigzag}}(r) = \left[\left(r + \frac{1}{2}\right) + \left(2C - 1 - r + \frac{1}{2}\right)\right] L_z^2 = 2C \cdot L_z^2$$

   Substitute $L_z = \frac{L_c}{2}$:

   $$\text{Work}_{\text{zigzag}}(r) = 2C \left(\frac{L_c}{2}\right)^2 = C \cdot \frac{1}{2} L_c^2 = \frac{s^2}{2C}$$

   Notice that **$r$ cancels completely**: every GPU $r \in \{0, \dots, C-1\}$ executes the exact same $\frac{s^2}{2C}$ causal pairs, raising utilization from $53.3\%$ to **$100.0\%$**!

??? question "How does DeepSpeed Ulysses differ from Ring Attention?"
    Both shard the sequence into $s/C$ tokens per GPU during MLP and norm layers, but Inside Attention **Ring Attention** keeps heads intact and circulates KV chunks around a P2P ring, whereas **DeepSpeed Ulysses** uses an `All-to-All` collective on $(Q, K, V)$ to transpose from *sequence-sharded, all-heads* $(s/C, h)$ to *full-sequence, head-sharded* $(s, h/C)$, runs standard local FlashAttention, and runs a second `All-to-All` on $O$. Ulysses has $O(1/C)$ communication volume per GPU but is capped at $C \le h_{kv}$ ($8$ on `Qwen3-0.6B`), whereas Ring Attention scales to arbitrary $C$ (and hybrid **USP** combines Ulysses intra-node with Ring Attention inter-node). See [Q&A: When should you use Ring Attention versus DeepSpeed Ulysses for Context Parallelism?](qa.md#when-should-you-use-ring-attention-versus-deepspeed-ulysses-for-context-parallelism).

---

### 3. Expert Parallelism (EP) & Node-Limited Routing

In a Mixture-of-Experts (MoE) layer (`Qwen3-235B-A22B` has $E = 128$ routed experts with $k = 8$ active per token; `DeepSeek-V3` has $E = 256$ routed experts with $k = 8$ active), each expert is an independent SwiGLU MLP.

**Expert Parallelism (EP)** places $E / P_{\text{EP}}$ whole experts on each of $P_{\text{EP}}$ GPUs:

1. **Dispatch (`All-to-All`):** Each GPU holds $B_{\text{local}} = b s$ tokens in BF16 ($2d$ bytes per token). The router selects $k$ experts per token. Assuming uniform load balance, a fraction $\frac{P_{\text{EP}} - 1}{P_{\text{EP}}}$ of those $k$ routed copies must travel over the network to other GPUs:
   $$\text{Bytes}_{\text{dispatch}} = 2 b s d \cdot k \cdot \frac{P_{\text{EP}} - 1}{P_{\text{EP}}}$$
2. **Expert MLP compute:** Each GPU runs its local $E / P_{\text{EP}}$ experts on the received tokens.
3. **Combine (`All-to-All`):** Each expert sends its $2d$-byte output back to the originating token's GPU:
   $$\text{Bytes}_{\text{combine}} = 2 b s d \cdot k \cdot \frac{P_{\text{EP}} - 1}{P_{\text{EP}}}$$

Total forward `All-to-All` communication per MoE layer is:

$$\text{Bytes}_{\text{EP, fwd}} = 4 b s d \cdot k \cdot \frac{P_{\text{EP}} - 1}{P_{\text{EP}}}$$

For $b s = 4{,}096$ tokens, $d = 2{,}048$, $k = 8$, and $P_{\text{EP}} = 8$:

$$\text{Bytes}_{\text{EP, fwd}} = 4 \times 4{,}096 \times 2{,}048 \times 8 \times \frac{7}{8} = 234{,}881{,}024\text{ B} = 224.0\text{ MiB per MoE layer}$$

In backward, both activation gradients and dispatch/combine paths reverse, doubling traffic to $448.0\text{ MiB}$ per MoE layer. To prevent `All-to-All` from saturating inter-node InfiniBand when $P_{\text{EP}} = 64$, **DeepSeek-V3** enforces **node-limited routing**: each token's $k = 8$ experts are restricted to at most $M = 4$ nodes, bounding inter-node IB traffic by $M$ regardless of $k$.

---

### 4. The 5D Parallelism Synthesis

| Dimension | Shards What | Primary Collective | Bandwidth Sensitivity | Best Scope |
| :--- | :--- | :--- | :--- | :--- |
| **TP + SP** | Weights & acts along $d$ / $s$ | `Reduce-Scatter` + `All-Gather` every layer | **Extreme** (on critical path) | Intra-node NVLink ($\le 8$ GPUs) |
| **CP** | All activations along $s$ | Ring P2P KV or Ulysses `All-to-All` | **Moderate** (overlaps if $s/C$ large) | Intra- or inter-node for $s \ge 32\text{K}$ |
| **EP** | MoE expert weights along $E$ | `All-to-All` dispatch & combine | **High** (mitigated by node-limiting / DualPipe) | Intra- + inter-node ($8$–$64$ GPUs) |
| **PP** | Layers along depth $L$ | Point-to-point $(b s d)$ at stage boundaries | **Low** (only layer boundaries) | Inter-node across racks |
| **DP / ZeRO** | Batch $B$ + states ($1/2/3$) | `Reduce-Scatter` (+ `All-Gather` in ZeRO-3) | **Low–Moderate** (overlaps with backward) | Outer loop across entire cluster |

```bash
uv run python book/code/context_parallel_balance.py
```

```text
Context Parallelism Causal Load Balance (unit = 0.5 * (s/C)^2 pairs)
------------------------------------------------------------------------
CP=4 Naive Ring   blocks/rank: [1.0, 3.0, 5.0, 7.0] | max/min=7.0x | util=57.1%
CP=4 Zig-Zag Ring blocks/rank: [4.0, 4.0, 4.0, 4.0] | max/min=1.0x | util=100.0%
CP=8 Naive Ring   blocks/rank: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0] | max/min=15.0x | util=53.3%
CP=8 Zig-Zag Ring blocks/rank: [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0] | max/min=1.0x | util=100.0%

MoE EP=8 All-to-All (4096 tok, d=2048, top-8): 224.0 MiB/layer
```

---

*Next: [Lecture 35: Low-Precision Training (FP8) & Modern Optimizers](35-fp8-training-and-optimizers.md)*

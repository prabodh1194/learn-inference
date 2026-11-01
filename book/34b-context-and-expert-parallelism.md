# 34b. Context parallelism, expert parallelism, and the 5D playbook

**Demo:** `book/code/context_parallel_balance.py`
**Moves:** raises causal Ring Attention cluster utilization from $53.3\%$ to $100.0\%$ via Zig-Zag pairing and bounds MoE `All-to-All` traffic
**Prereq:** [23. MoE and expert parallelism](23-moe-and-expert-parallelism.md), [34. TP + SP, Pipeline schedules, and 3D parallelism](34-3d-parallelism-and-schedules.md)

---

## The problem

Lecture 34 showed how Tensor Parallelism (TP), Sequence Parallelism (SP), and
Pipeline Parallelism (PP) split weights and saved tensors across GPUs. Yet two
workloads still break 3D parallelism.

First, consider **long-context training** at $s = 32\text{K}$ to $1\text{M}$
tokens. In Sequence Parallelism (Lecture 34), tokens are split along $s$ only in
the norm and residual layers; right before the `QKV` projections, an `all-gather`
rebuilds the full sequence of length $s$ on every GPU inside the TP group. Worse,
TP cannot scale past the $t = 8$ GPUs inside one NVLink node, so a million-token
sequence still overflows VRAM inside attention.

Second, consider **Mixture-of-Experts (MoE) training** (`Qwen3-235B-A22B` or
`DeepSeek-V3`). Replicating $128$ to $256$ experts on every data-parallel rank
blows up memory, while slicing each tiny expert MLP with TP adds per-layer
`all-reduce` latency for very little compute.

We need two more sharding axes: **Context Parallelism (CP)** to split the
sequence inside attention itself, and **Expert Parallelism (EP)** to place whole
experts on separate GPUs. Together with DP, TP/SP, and PP, they form the
**5D parallelism** playbook.

---

## See it

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

Look at the `CP=8` rows. Splitting a causal sequence into 8 contiguous chunks
leaves GPU 7 doing **$15\times$** the attention work of GPU 0, so the 8-GPU ring
runs at only **53.3% utilization**. Pairing symmetric front-and-back sub-chunks
(**Zig-Zag Ring Attention**) gives every GPU the exact same workload (`8.0`
units) and restores **100.0% utilization**.

---

## The idea

### 1. Context Parallelism and Ring Attention

**Context Parallelism (CP)** splits the input sequence of length $s$ into $C$
chunks of length $L_c = s / C$ over $C$ GPUs **everywhere in the network**,
including inside self-attention.

Why is attention the only place CP needs to communicate? Every linear projection
(`Q, K, V, O`, `gate, up, down`), RMSNorm, and SwiGLU acts on each token
independently. Those layers need all $d$ hidden channels of a token, never its
neighboring tokens, so they run on a local chunk of $L_c = s / C$ tokens with
**zero network traffic**. Self-attention is the one exception, because query
tokens $Q_r$ on GPU $r$ must attend to key-value pairs $(K_j, V_j)$ sitting on
other GPUs $j \neq r$.

**Ring Attention (Liu et al., 2023)** hides that KV transfer underneath the
attention math using FlashAttention's online softmax (Lecture 17). Arrange the
$C$ GPUs in a logical ring $0 \to 1 \to 2 \to \dots \to C-1 \to 0$. In step 0,
GPU $r$ computes attention between its local queries $Q_r$ and its own keys and
values $(K_r, V_r)$. While the tensor cores run that matmul, the network card
sends $(K_r, V_r)$ to neighbor $(r+1) \bmod C$ and receives the next KV chunk
from $(r-1) \bmod C$. Over $C$ ring steps, each KV chunk walks all the way
around the ring while GPU $r$ updates its running online-softmax state
$(m_r, \ell_r, O_r)$ in place.

When does the math completely hide the wire transfer? Let us derive the
compute-to-communication ratio for one ring step on chunk length $L_c = s/C$,
with $h_q$ query heads and $h_{kv}$ KV heads of width $d_h$ (so
$d_q = h_q d_h$ and $d_{kv} = h_{kv} d_h$).

An unmasked forward attention block between $L_c$ queries and $L_c$ keys runs
two matmuls ($Q K^\top$ and $P V$), which takes $4 b L_c^2 d_q$ FLOPs.

$$\text{FLOPs}_{\text{step}} = 4 b L_c^2 d_q = \frac{4 b s^2 d_q}{C^2}$$

Meanwhile, sending both $K$ and $V$ of length $L_c$ in BF16 (2 bytes per
element) to the next GPU in the ring moves $4 b L_c d_{kv}$ bytes.

$$\text{Bytes}_{\text{step}} = 2 \times 2 b L_c d_{kv} = \frac{4 b s d_{kv}}{C}$$

Divide the FLOPs per step by the bytes sent per step, and cancel $4 b$ and
$d_h$:

$$\frac{\text{FLOPs}_{\text{step}}}{\text{Bytes}_{\text{step}}} = \frac{4 b s^2 d_q / C^2}{4 b s d_{kv} / C} = \frac{s}{C} \cdot \frac{h_q}{h_{kv}} = L_c \cdot g_{\text{GQA}}$$

Here $g_{\text{GQA}} = h_q / h_{kv}$ is the Grouped-Query Attention group size
($16 / 8 = 2$ on `Qwen3-0.6B`). As long as the local chunk length $L_c = s/C$
satisfies $L_c \cdot g_{\text{GQA}} > \frac{\text{Peak TFLOP/s}}{\text{Link GB/s}}$,
every P2P KV transfer finishes before the attention kernel on the current chunk
ends.

---

### 2. Why naive causal Ring Attention wastes half your GPUs, and how Zig-Zag fixes it

Suppose we split a causal sequence of length $s$ into $C$ contiguous chunks so
GPU $r \in \{0, 1, \dots, C-1\}$ holds token positions $[r L_c,\; (r+1) L_c)$.
Because causal attention only lets query $q$ read keys $k \le q$, early tokens
in the sequence have almost nothing to attend to, while late tokens must attend
to the entire prompt.

We can count the exact number of valid $(q, k)$ pairs evaluated by GPU $r$ over
all $C$ ring steps. First, GPU $r$'s queries sit after all $r$ earlier chunks
$j \in \{0, \dots, r-1\}$, so they attend to every token in those $r$ chunks,
which is $r L_c^2$ pairs. Second, within GPU $r$'s own chunk $j = r$, causal
masking keeps only the lower triangle of the $L_c \times L_c$ block, adding
$\frac{1}{2} L_c^2$ pairs. Third, all later chunks $j > r$ sit in the future, so
they contribute zero pairs.

Add those pieces to get the total causal work on GPU $r$.

$$\text{Work}_{\text{naive}}(r) = \left(r + \frac{1}{2}\right) L_c^2 = (2r + 1) \cdot \frac{1}{2} L_c^2$$

Compare the straggler GPU $C-1$ (holding the end of the sequence) against GPU 0
(holding the start of the sequence).

$$\frac{\text{Work}_{\text{naive}}(C-1)}{\text{Work}_{\text{naive}}(0)} = \frac{2(C-1) + 1}{2(0) + 1} = 2C - 1$$

Because all GPUs must wait for the slowest rank $C-1$ to finish, cluster
utilization is the average work over all $C$ GPUs divided by the max work on
GPU $C-1$. Using $\sum_{r=0}^{C-1} (2r + 1) = C^2$, we get the utilization fraction.

$$\text{Util}_{\text{naive}}(C) = \frac{\frac{1}{C}\sum_{r=0}^{C-1} (2r+1)}{2C - 1} = \frac{C^2 / C}{2C - 1} = \frac{C}{2C - 1}$$

At $C = 4$, GPU 3 does **$7\times$** the work of GPU 0, and utilization is only
$4/7 = 57.1\%$. At $C = 8$, GPU 7 does **$15\times$** the work of GPU 0, and
utilization drops to $8/15 = 53.3\%$. Nearly half your GPUs sit idle waiting on
the tail of the sequence.

How do we balance a triangle over $C$ workers? Pair a tiny slice from the top
of the triangle with a wide slice from the bottom. Instead of $C$ contiguous
chunks of length $L_c = s/C$, **Zig-Zag Ring Attention** cuts the sequence into
**$2C$ sub-chunks** of half length $L_z = \frac{s}{2C} = \frac{1}{2} L_c$,
indexed $0, 1, \dots, 2C-1$. We then give GPU $r$ one sub-chunk from the front
and its mirror image from the back.

$$\text{Sub-chunks}(r) = \{\, r, \;\; 2C - 1 - r \,\}$$

For $C = 4$ (8 sub-chunks $0 \dots 7$), GPU 0 gets $\{0, 7\}$, GPU 1 gets
$\{1, 6\}$, GPU 2 gets $\{2, 5\}$, and GPU 3 gets $\{3, 4\}$.

Now sum the causal work of both sub-chunks on GPU $r$. The early sub-chunk
$a = r$ evaluates $\left(r + \frac{1}{2}\right) L_z^2$ causal pairs, while the
late sub-chunk $b = 2C - 1 - r$ evaluates
$\left(2C - 1 - r + \frac{1}{2}\right) L_z^2$ causal pairs. Add them together:

$$\text{Work}_{\text{zigzag}}(r) = \left[\left(r + \frac{1}{2}\right) + \left(2C - 1 - r + \frac{1}{2}\right)\right] L_z^2 = 2C \cdot L_z^2$$

Substitute $L_z = \frac{L_c}{2} = \frac{s}{2C}$ into that sum:

$$\text{Work}_{\text{zigzag}}(r) = 2C \left(\frac{L_c}{2}\right)^2 = C \cdot \frac{1}{2} L_c^2 = \frac{s^2}{2C}$$

Look at the final expression: **the rank index $r$ cancels out completely**.
Every GPU $r \in \{0, \dots, C-1\}$ computes the exact same $\frac{s^2}{2C}$
causal pairs, bringing attention utilization from $53.3\%$ to **$100.0\%$**.

??? question "How does DeepSpeed Ulysses differ from Ring Attention?"
    Both keep $s/C$ tokens per GPU in the MLP and norm layers, but they handle
    attention differently. **Ring Attention** keeps all heads on each GPU and
    passes KV chunks around a P2P ring. **DeepSpeed Ulysses** instead runs an
    `All-to-All` collective on $(Q, K, V)$ to transpose the layout from
    sequence-sharded $(s/C, h)$ to head-sharded $(s, h/C)$, runs local
    FlashAttention over the full sequence for $h/C$ heads, and runs a second
    `All-to-All` on the output $O$. Ulysses sends fewer bytes per GPU as $C$
    grows, but it cannot scale past the number of KV heads ($C \le h_{kv} = 8$
    on `Qwen3-0.6B`), whereas Ring Attention scales to any $C$. Hybrid **USP**
    uses Ulysses inside the node and Zig-Zag Ring Attention between nodes.
    [Full answer](qa.md#when-should-you-use-ring-attention-versus-deepspeed-ulysses-for-context-parallelism)

---

### 3. Expert Parallelism (EP) and node-limited routing

In a Mixture-of-Experts layer (`Qwen3-235B-A22B` has $E = 128$ routed experts
with $k = 8$ active per token; `DeepSeek-V3` has $E = 256$ with $k = 8$ active),
each expert is an independent SwiGLU MLP. Because only $k$ of the $E$ experts
fire for any single token, **Expert Parallelism (EP)** places $E / P_{\text{EP}}$
whole experts on each of $P_{\text{EP}}$ GPUs and ships tokens to whichever GPUs
own their chosen experts.

Walk one MoE layer forward pass on $B_{\text{local}} = b s$ local tokens in
BF16 ($2d$ bytes per token). First comes the **dispatch `All-to-All`**: the
router picks $k$ experts per token, which produces $k \cdot b s$ routed token
copies. When load is balanced over the $P_{\text{EP}}$ GPUs, a fraction
$\frac{P_{\text{EP}} - 1}{P_{\text{EP}}}$ of those copies belong to experts on
remote GPUs and must cross the network.

$$\text{Bytes}_{\text{dispatch}} = 2 b s d \cdot k \cdot \frac{P_{\text{EP}} - 1}{P_{\text{EP}}}$$

Second, each GPU runs its local $E / P_{\text{EP}}$ expert MLPs on the tokens it
received. Third comes the **combine `All-to-All`**: each expert sends its
$2d$-byte output vector back to the GPU that originated the token, moving the
same volume a second time.

$$\text{Bytes}_{\text{combine}} = 2 b s d \cdot k \cdot \frac{P_{\text{EP}} - 1}{P_{\text{EP}}}$$

Add dispatch and combine to get total forward `All-to-All` traffic per MoE layer.

$$\text{Bytes}_{\text{EP, fwd}} = 4 b s d \cdot k \cdot \frac{P_{\text{EP}} - 1}{P_{\text{EP}}}$$

Plug in $b s = 4{,}096$ tokens, hidden width $d = 2{,}048$, top-$k = 8$, and
$P_{\text{EP}} = 8$ GPUs:

$$\text{Bytes}_{\text{EP, fwd}} = 4 \times 4{,}096 \times 2{,}048 \times 8 \times \frac{7}{8} = 234{,}881{,}024\text{ B} = 224.0\text{ MiB per MoE layer}$$

During backward, gradients return via both `All-to-All` steps, doubling the
traffic to $448.0\text{ MiB}$ per MoE layer. When $P_{\text{EP}} = 64$ spans 8
nodes, an unrestricted `All-to-All` would spray all $k = 8$ copies to 8
different nodes over slower InfiniBand. **DeepSeek-V3** prevents that bottleneck
with **node-limited routing**: the router restricts each token's $k = 8$ chosen
experts to sit on at most $M = 4$ nodes, capping inter-node traffic while
DualPipe (Lecture 34) hides the transfer behind compute.

---

### 4. Putting the 5D parallelism playbook together

Each of the five parallelism dimensions cuts a different axis of the training
ledger and belongs on a specific wire tier:

| Dimension | Shards what | Primary collective | Bandwidth sensitivity | Best hardware scope |
| :--- | :--- | :--- | :--- | :--- |
| **TP + SP** | Weights and activations along $d$ and $s$ | `Reduce-Scatter` + `All-Gather` every layer | Extreme (on critical path) | Intra-node NVLink ($\le 8$ GPUs) |
| **CP** | All activations along $s$ | Ring P2P KV or Ulysses `All-to-All` | Moderate (overlaps when $s/C$ is large) | Intra- or inter-node for $s \ge 32\text{K}$ |
| **EP** | MoE expert weights along $E$ | `All-to-All` dispatch and combine | High (tamed by node-limiting and DualPipe) | Intra- and inter-node ($8$ to $64$ GPUs) |
| **PP** | Layers along depth $L$ | Point-to-point $(b s d)$ at stage boundaries | Low (only at stage boundaries) | Inter-node across racks |
| **DP / ZeRO** | Batch $B$ and model state | `Reduce-Scatter` (+ `All-Gather` in ZeRO-3) | Low to moderate (overlaps with backward) | Outer ring over the whole cluster |

---

## What you should see

Running `book/code/context_parallel_balance.py` confirms that naive contiguous
chunking creates a $(2C - 1)\times$ causal work imbalance across $C$ GPUs
($15\times$ at $C = 8$, wasting $46.7\%$ of cluster attention time), while
Zig-Zag symmetric pairing gives every rank the exact same $\frac{s^2}{2C}$ work
and $100\%$ utilization.

---

## Go deeper

- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (Liu et al., 2023), overlapping P2P KV transfers with blockwise online softmax.
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509) (Jacobs et al., 2023), `All-to-All` sequence-to-head transposition.
- [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Nouamane Tazi et al., 2025), benchmarks across 5D parallelism configurations.

---

## Check yourself

1. Why do MLP and RMSNorm layers need zero communication under Context
   Parallelism?
2. Derive the $(2C - 1)$ max-to-min work ratio of naive contiguous causal Ring
   Attention, and show why pairing sub-chunks $\{r, 2C - 1 - r\}$ makes the work
   independent of $r$.
3. Why is DeepSpeed Ulysses limited to $C \le h_{kv}$ GPUs while Ring Attention
   can scale to larger $C$?

---

## Next

**[35. Low-precision training (FP8) and modern optimizers](35-fp8-training-and-optimizers.md)**:
double tensor-core GEMM throughput with $1 \times 128$ FP8 tile scaling, replace
auxiliary MoE balance losses with a bias thermostat, and orthogonalize updates
with Muon.

```bash
uv run python book/code/fp8_trainingsim.py
```

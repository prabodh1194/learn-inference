# 23. MoE and expert parallelism

**Build:** `jaxlm/moe.py`, routing and a small MoE layer
**Test:** `tests/test_23_moe.py` · **Moves:** understanding of why frontier models are shaped this way
**Prereq:** [22. Tensor parallelism](22-tensor-parallelism.md)

---

## The problem

Every model in this book so far is **dense**: every parameter participates in every
token. Doubling parameters doubles the compute per token, and, since decode is
memory-bound, doubles the bytes you must move per token.

That's a hard ceiling on capacity. Mixture of Experts breaks it.

Most frontier open models are now MoE (DeepSeek, Qwen's larger models, Mixtral,
GPT-OSS). Understanding them is no longer optional for inference work.

---

## The idea

Replace each MLP block with **many** MLPs ("experts") plus a **router** that
activates only a few per token. The router is a small scoring layer: it gives
every expert a score for the current token, and activates the few with the
highest scores (**top-k**; top-2 means the two best, top-1 the single best):

```
dense:  token -> MLP (all params)              -> output
MoE:    token -> router -> pick top-2 of 128   -> output
```

With 128 experts and top-2 routing you get the *capacity* of 128 MLPs at roughly
the *compute* of 2:

```
2 / 128  =  0.0156  =  1.56%  of dense compute per token
```

That 0.0156 rounds to 0.016, which is the 1.6% you'll see quoted. Now read
the memory side of the same sentence: you still store all 128 expert weight
matrices, so the *memory* is unchanged, 100% of the dense model's. Compute
falls to ~1.6%, memory stays at 100%. That asymmetry is the whole story of
this lecture, and the next section names its two halves.

The vocabulary that follows from this, and which trips people up:

- **Total parameters**: everything stored. Determines memory.
- **Active parameters**: what runs per token. Determines compute.

DeepSeek-V3 is 671B total, 37B active. Its MoE shape is **1 shared expert plus
256 routed experts, top-8**: each token's router picks the 8 best of 256
(`8/256 = 3.1%` of the experts), on top of the always-on shared one. The
original DeepSeekMoE made the same point at laptop scale: a **16B-total,
2.8B-active** model going head-to-head with a 7B dense LLaMA while doing only
`2.8/7 = 40%` of the arithmetic per token (and running `7/2.8 = 2.5×` faster).
With the book's standing convention of 2 bytes per weight (the same arithmetic
Lecture 22 used for its 70B example):

```
total :  671B × 2 B  =  1,342 GB  ≈  1.34 TB    must be resident, always
active:   37B × 2 B  =     74 GB                read per token, at most
```

**You must hold all 671B in memory even
though each token touches 37B**, because the router might select any expert.
The 1.34 TB is what you buy; the 74 GB is what you actually use per token, an
18.1× buffer (`1,342 / 74 = 18.1`, the same ratio as `671 / 37`).

That mismatch is the entire inference story for MoE.

The small model makes the trap concrete: DeepSeekMoE's 16B model fits a 40 GB
GPU because `16B × 2 B = 32 GB`, *not* because only 2.8B is active. Believe the
second reason and you'll wrongly conclude a 100B-total MoE fits a 40 GB card
"because only 20B is active". Memory is set by **total**; active only sets
compute.

### What this does to your bottlenecks

Go back to Lecture 02 and re-derive it.

**Compute per token falls**: only active experts run.

**Memory capacity requirement stays enormous**: all weights resident.

**Memory *bandwidth* per token falls**, you only read the active experts.

Now the re-derivation, with the DeepSeek-V3 numbers, using the book's simple
model of one forward pass (2 FLOPs per parameter per token, 2 bytes per
weight, from Lecture 02). Compare the MoE to a hypothetical **dense** model of
the same 671B total size:

```
                  dense twin (671B)          MoE (671B stored, 37B active)
bytes/token:      671B × 2 B  = 1,342 GB      37B × 2 B  =  74 GB
compute/token:    2 × 671B    = 1,342 GFLOP   2 × 37B    =  74 GFLOP
intensity:        1,342/1,342 = 1 op:byte     74/74      =  1 op:byte
```

The sink of it: the MoE moves **18.1× fewer bytes** (`1,342 GB / 74 GB`) and
does 18.1× less compute, so the *ratio* of compute to bytes barely moves.
Arithmetic intensity is roughly *unchanged*: compute and bytes both fall by the
same active/total factor (37/671, about 5.5%), so you haven't moved along the
roofline. What falls is the **absolute** bytes per token, and decode latency is
made of absolute bytes: the MoE's 74 GB per token is why a 671B-parameter model
can decode like a 37B one. MoE buys capacity at nearly fixed decode cost; it
does not buy you a better roofline position.

(One honesty note on that table: it counts weights only, the dominant term.
Both models also read the same KV cache bytes per token, identical in the two
columns, and that shared term is exactly what makes the change "roughly"
rather than exactly zero: with a 2048-token context it nudges the MoE's
intensity from 1.000 to about 0.997, still essentially 1.)

But a new cost appears: **routing is dynamic and unbalanced.** Which experts a
token needs isn't known until the router runs, and different tokens in a batch
want different experts.

### Fine-grained and shared experts

Two design choices explain *why* 256 experts rather than 8, and why a couple are
"shared".

**Fine-grained segmentation** slices each fat expert into smaller ones and routes
to more of them: 16 experts with top-2 becomes 64 with top-8. Same total
parameters, same active parameters, same FLOPs — but the router has far more
combinations to choose from:

```
C(16, 2)  =  16·15 / 2          =  120        possible expert pairs
C(64, 8)  =  64! / (8! 56!)     ≈  4.4 billion
```

Granularity buys *combinations*, not capacity: the router can assign each token a
more precisely matched subset. That is what the "256 skinny experts" in the V3
number above is doing.

**Shared experts** are a couple of experts that are always on, no routing. They
hold the knowledge every token needs (grammar, common facts); the routed experts
hold the specialist edge. The serving consequence matters: shared experts are
*dense* compute — every token hits them, so they batch like a dense model —
while routed experts are *sparse* dispatch, tokens scattering to different GPUs.
V3's "1 shared + 256 routed" is exactly this split.

The motivation for both is that knowledge in a dense model is **hybrid** (each
neuron mixes many concepts) and **redundant** (one concept in many places).
Fine-grained + shared is the fix: isolate the common knowledge in always-on
experts, and let the routed ones specialize.

### Expert parallelism

TP splits every tensor across GPUs. EP takes a different axis: **put whole experts
on different GPUs.**

```
TP:  every GPU holds a slice of every expert   -> all-reduce per layer
EP:  each GPU holds complete experts           -> route tokens to the right GPU
```

Under EP the tokens themselves travel. An **all-to-all** is the other classic
collective (the family Lecture 21 named): every GPU hands every other GPU the
tokens that chose its experts, all in one coordinated exchange. Unlike the
all-reduce, where each GPU ends with a copy of a shared total, here each GPU
ends with *different* data: the tokens it must process.

The tradeoff, and it's a clean one:

| | TP | EP |
|---|---|---|
| Communication | all-reduce **every layer** | all-to-all token routing |
| Volume | proportional to hidden size | proportional to routed tokens |
| Optimizes | **latency** | **throughput** |
| Scales to | within a node | across nodes |

Why the volume column differs: TP's all-reduce carries a full-sized
activation, the whole hidden dimension, at *every* layer, every token, every
step, in both directions. EP's all-to-all carries only the tokens that happen
to switch GPUs, and each carries no more than one hidden vector. Most tokens in
a batch are usually routed to experts on their own GPU and never travel at all;
only the minority cross the interconnect. Fewer bytes, less often: that's why
EP survives lower-bandwidth interconnects and scales multi-node where TP
doesn't, and why its cost is a *throughput* question (how many tokens can the
layers keep flowing) rather than a per-token latency one.

**Production deployments mix them**: TP for attention (which isn't expert-sharded)
and EP for the MoE layers. Kiely's Fig 5.15 (p.146) diagrams exactly this.

One caveat from the largest deployments: cross-node EP's all-to-all is *not*
free — DeepSeek reports the compute-to-communication ratio is near 1:1 across
nodes, which is why V3 shipped custom all-to-all kernels and a "DualPipe" to
hide the transfer underneath compute. "EP scales multi-node" is true, but only
with that overlap work. And with 257 experts to shard, V3 **skipped tensor
parallelism entirely**: enough experts give EP the intra-node sharding TP would,
without the per-layer all-reduce. TP is the default when experts are few; EP can
absorb its role when they are many.

### Load imbalance is the operational problem

Nothing guarantees experts get equal traffic. Some are popular; their GPU becomes
the bottleneck while others idle. An expert is a GPU's local workload, after
all: a hot expert saturates its GPU while a cold one leaves its GPU mostly
empty, and the empties cannot help the busy one.

Mitigations you'll see in the wild:

- **Capacity factor**: a cap on how many tokens one expert will accept in a
  single step, quoted as a multiple of its fair share (each expert getting
  `1/n_experts` of the batch). Tokens past the cap are dropped or rerouted.
- **Auxiliary load-balancing loss**: a training-time nudge toward uniformity; a
  small penalty term added when routing gets lopsided, so the router learns to
  share work before serving ever sees it.
- **Auxiliary-loss-free routing** (DeepSeek-V3's choice): no penalty term at all.
  A per-expert bias is added to the router's scores *at serving time* —
  under-used experts get their bias nudged up, over-used nudged down — so the
  router balances load without fighting the next-token objective. No gradient,
  no loss term; the language-modeling loss stays pure, and the bias adapts live.
- **Expert replication**: duplicate hot experts across GPUs, buying the
  popular ones extra attention at the cost of memory.

This is an inference-time reality even though its main lever is at training time,
and it's why MoE serving has more variance than dense serving.

---

## Build it

Small scale: the concepts, not a production MoE:

1. Implement a router (top-k over a linear projection) and a few small experts in
   `jaxlm/moe.py`.
2. `uv run pytest tests/test_23_moe.py -v`, verify top-k selection and that the
   output matches a dense-equivalent reference for a single expert.
3. **Measure routing distribution** on real token sequences. Plot tokens per
   expert. Even with a randomly initialized router, note how uneven it is.
4. Shard experts across devices with `jax.sharding` and observe the all-to-all in
   the HLO: the EP analogue of Lecture 22's all-reduce.
5. **Compute the arithmetic intensity** of an MoE layer versus a dense layer of
   the same total parameter count. That number is the whole argument for MoE.

---

## What you should see

**Compute per token far below what total parameters suggest.**

**Memory unchanged**: you still hold everything.

**Uneven expert utilization**, even on toy data.

**All-to-all instead of all-reduce** in the sharded HLO, a different
communication pattern with different scaling behaviour.

---

## Go deeper

- **[Switch Transformers](https://arxiv.org/abs/2101.03961)** (Fedus et al.),   top-1 routing, capacity factors, load balancing. The clearest introduction.
- **[Mixtral of Experts](https://arxiv.org/abs/2401.04088)**: a real open MoE with
  inference details.
- **[DeepSeek-V3](https://arxiv.org/abs/2412.19437)**: fine-grained experts and
  shared experts; the current state of the art in MoE inference design.
- **Kiely §2.2.4** (p.53), MoE architecture.
- **Kiely §5.4.2** (p.145) and **Fig 5.15** (p.146), EP for throughput, and the
  TP+EP mixed deployment.

---

## Check yourself

1. DeepSeek-V3 (671B total, 37B active): how much VRAM to serve it, and how much
   compute per token? Why is the answer to the first not "37B worth"?
2. MoE moves far fewer bytes per token than a dense model of the same total size,
   yet its arithmetic intensity is about the same. Explain both.
3. TP optimizes latency, EP optimizes throughput. Why does that follow from their
   communication patterns?
4. Why does load imbalance hurt more in EP than in TP?
5. You must serve an MoE across two nodes with slow interconnect. TP, EP, or both,
   and why?

---

**Part IV complete.** You've seen sharding declared and derived, then written by
hand, and understood the two axes models are split along.

## Next

**[24. Serving](24-serving.md)**: Part V turns your engine into a service.

Back to the laptop: most of Part V is systems work rather than kernels, and the
tests fake the engine so they run without a GPU.

The architectural point of L24 is why vLLM runs its API server in a **separate
process**, HTTP handling and tokenization on the engine loop steal time from
the scheduler.

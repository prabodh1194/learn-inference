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
activates only a few per token:

```
dense:  token -> MLP (all params)              -> output
MoE:    token -> router -> pick top-2 of 128   -> output
```

With 128 experts and top-2 routing you get the *capacity* of 128 MLPs at roughly
the *compute* of 2.

The vocabulary that follows from this, and which trips people up:

- **Total parameters**: everything stored. Determines memory.
- **Active parameters**: what runs per token. Determines compute.

DeepSeek-V3 is 671B total, 37B active. **You must hold all 671B in memory even
though each token touches 37B**, because the router might select any expert.

That mismatch is the entire inference story for MoE.

### What this does to your bottlenecks

Go back to Lecture 02 and re-derive it.

**Compute per token falls**: only active experts run.

**Memory capacity requirement stays enormous**: all weights resident.

**Memory *bandwidth* per token falls**, you only read the active experts.

Note what this does **not** say. Arithmetic intensity is roughly *unchanged*,
compute and bytes both fall by the same active/total factor, so you haven't moved
along the roofline. What falls is the **absolute** bytes per token, and decode
latency is made of absolute bytes. MoE buys capacity at nearly fixed decode cost;
it does not buy you a better roofline position.

But a new cost appears: **routing is dynamic and unbalanced.** Which experts a
token needs isn't known until the router runs, and different tokens in a batch
want different experts.

### Expert parallelism

TP splits every tensor across GPUs. EP takes a different axis: **put whole experts
on different GPUs.**

```
TP:  every GPU holds a slice of every expert   -> all-reduce per layer
EP:  each GPU holds complete experts           -> route tokens to the right GPU
```

The tradeoff, and it's a clean one:

| | TP | EP |
|---|---|---|
| Communication | all-reduce **every layer** | all-to-all token routing |
| Volume | proportional to hidden size | proportional to routed tokens |
| Optimizes | **latency** | **throughput** |
| Scales to | within a node | across nodes |

EP moves less data (routed tokens are smaller than full activations) which is
why it survives lower-bandwidth interconnects and scales multi-node where TP
doesn't.

**Production deployments mix them**: TP for attention (which isn't expert-sharded)
and EP for the MoE layers. Kiely's Fig 5.15 (p.146) diagrams exactly this.

### Load imbalance is the operational problem

Nothing guarantees experts get equal traffic. Some are popular; their GPU becomes
the bottleneck while others idle.

Mitigations you'll see in the wild:

- **Capacity factor**: cap tokens per expert; drop or reroute the overflow.
- **Auxiliary load-balancing loss**: a training-time nudge toward uniformity.
- **Expert replication**: duplicate hot experts across GPUs.

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

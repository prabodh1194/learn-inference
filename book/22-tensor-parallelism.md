# 22 — Tensor parallelism

**Build:** `jaxlm/sharding.py`, then TP by hand in `engine/`
**Test:** `tests/test_22_tp.py` (cuda, multi-GPU) · **Moves:** per-user latency — and reveals where scaling stops
**Prereq:** [21 — JAX and XLA](21-jax-and-xla.md)

> **Two or more GPUs required.** On Vast.ai, a 2×3090 box. Sharding *logic* can be
> reasoned about on one device; scaling curves cannot.

---

## The problem

Two reasons to use more than one GPU:

**The model doesn't fit.** 70B in FP16 is 140GB. No single GPU holds it.

**One GPU is too slow for one user.** Decode is memory-bound (Lecture 02), so a
single user's tokens/sec is capped by *one* GPU's bandwidth. Batching raises
aggregate throughput but never helps that one user. Splitting the weights across
GPUs multiplies the bandwidth available to a single sequence.

That second reason is why TP is a **latency** optimization, and it's the one people
miss.

---

## The idea

Three ways to split a model:

| | Splits | Cost | Use |
|---|---|---|---|
| **Pipeline (PP)** | layers across GPUs | bubbles; poor latency | multi-node, low bandwidth |
| **Tensor (TP)** | tensors *within* each layer | all-reduce every layer | **default within a node** |
| **Expert (EP)** | MoE experts across GPUs | token routing | MoE throughput (L23) |

TP is the default for single-node inference because every GPU works on *every*
token — no pipeline bubbles, and latency genuinely drops.

### How the split works

The elegance is that the two matmuls in an MLP shard complementarily:

```
Column-parallel (first matmul):  split weights by COLUMN
    each GPU computes part of the hidden dimension
    -> no communication needed

Row-parallel (second matmul):    split weights by ROW
    each GPU computes a PARTIAL sum of the output
    -> all-reduce to combine
```

Column-then-row means **one all-reduce per MLP block**, not two. Attention shards
the same way, by heads: each GPU owns a subset of heads, then all-reduce after the
output projection.

So: **two all-reduces per transformer layer.** For a 28-layer model, 56 collectives
per forward pass. That's the cost, and it's why interconnect matters.

### Why scaling isn't linear

Compute per GPU divides by N. Communication does not — an all-reduce moves data
proportional to the *hidden size*, regardless of how many GPUs share it.

```
time = compute/N + communication(N)
```

Past some N, communication dominates and adding GPUs stops helping. Where that
happens depends on interconnect:

| Link | Bandwidth |
|---|---|
| NVLink | ~900 GB/s |
| PCIe 5 | ~125 GB/s |

**A 7× difference in the thing that isn't parallelized.** This is why the same
model scales beautifully on one box and poorly on another with identical GPUs.

### Where the rules of thumb break

Conventional guidance says: low interconnect bandwidth → use pipeline parallelism
instead of TP.

The [field notes](field-notes.md) record an operator with **2× GH200 and no
NVLink** (PCIe only, 125 GB/s instead of 900) who followed exactly that advice.
**Pipeline parallel lost. TP2 won.** On the hardware profile where guides say it
shouldn't.

The same operator found `--max-num-seqs 16` — a scheduler concurrency limit from
Lecture 08 — mattered more than the parallelism strategy did.

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

Now you're placing the collective yourself — in the position XLA chose for you.
Compare against nano-vllm's `layers/linear.py`, which does exactly this.

---

## Build it

1. Shard the JAX model with `NamedSharding`. Verify output matches single-device.
2. Find the inserted all-reduce in the HLO.
3. Implement `ColumnParallelLinear` / `RowParallelLinear` in PyTorch with NCCL.
4. Verify TP output matches single-GPU output exactly.
5. **Measure the scaling curve** at 1, 2, 4 GPUs — plot with
   `bench/plot.py::scaling`, which takes measured throughput against ideal linear.
6. Find where it goes sublinear, and check `nvidia-smi topo -m` to see whether
   you have NVLink or PCIe.

**Predict first:** what speedup at 2 GPUs? At 4? Write both down before measuring.

---

## What you should see

**Sublinear scaling.** 2 GPUs give well under 2×. The gap is communication.

**Better scaling with NVLink** than PCIe, on identical GPUs.

**Better scaling on larger models** — more compute per layer amortizes the same
all-reduce.

**Worse scaling at small batch sizes** — less compute to hide the collective
behind.

**Latency improves; throughput per GPU falls.** You're spending hardware to make
one user faster. That's the trade, and it's the right one only sometimes.

---

## Go deeper

- **[Megatron-LM](https://arxiv.org/abs/1909.08053)** (Shoeybi et al.) — §3 has
  the column/row split you just implemented. The original.
- **Kiely §5.4–5.4.1** (p.142–145) — TP/PP/EP compared, and TP for latency.
- **Kiely §5.4.3** (p.146) — multi-node, where TP stops being the answer.
- **nano-vllm `nanovllm/layers/linear.py`** — a readable production TP
  implementation.
- **[Field notes](field-notes.md)** — the GH200 case where PP lost to TP without
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

**Next:** [23 — MoE and expert parallelism](23-moe-and-expert-parallelism.md) — a
different axis to split along.

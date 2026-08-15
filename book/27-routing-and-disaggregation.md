# 27. Routing and disaggregation

**Build:** `serve/router.py`, `serve/disaggregated.py` · **Test:** `tests/test_27_routing.py`
**Moves:** cache hit rate across replicas; TTFT/TPOT interference
**Prereq:** [26. Versus vLLM](26-versus-vllm.md)

---

## The problem

A **replica** is a full copy of your service: its own machine, its own GPU, its
own copy of the model, and its own scratchpad. One replica has a ceiling, you
found it in Lecture 25. Past that you add replicas, and two new problems appear
that don't exist on a single box.

**Your prefix cache fragments.** The scratchpad is the KV cache from Lecture 05,
the running notes the model keeps about each conversation, and each replica
keeps a private copy. **Round-robin** means taking requests in turn: replica 1,
2, 3, 1, 2, 3. It balances load perfectly but treats every request as a
stranger. It sends a user's follow-up to a replica that has never seen their
conversation, and every turn is a cache miss. Lecture 10's win evaporates
precisely when you scale.

**Prefill and decode still interfere.** Chunked prefill (Lecture 11) softened this,
but they remain fundamentally different workloads, compute-bound versus
memory-bound, competing for one GPU.

---

## Part 1: Cache-aware routing

Round-robin balances *load*. It ignores *state*, and the KV cache is state.

```
round-robin:   user's turn 2 -> replica 3 (cold)   -> full prefill
cache-aware:   user's turn 2 -> replica 1 (warm)   -> prefix hit
```

Route by **where the prefix already lives**. The **prefix hash** is a
fingerprint of the conversation's tokens, the same identity Lecture 10's block
hashes used; a replica keeps a list of the fingerprints it holds, so "have you
seen this conversation?" becomes a dictionary lookup. **Load** is how much work
a replica has right now, in-flight requests, queue depth, whichever simple
proxy you trust:

```python
def route(request, replicas):
    prefix_hash = hash_prefix(request.prompt_ids)
    warm = [r for r in replicas if r.has_prefix(prefix_hash)]
    if warm:
        return min(warm, key=lambda r: r.load)   # warmest AND least loaded
    return min(replicas, key=lambda r: r.load)
```

The tension is immediate and unavoidable: **the replica with your cache may be the
busy one.** Route purely by cache affinity and you create hotspots, one replica
drowning while the others idle; route purely by
load and you lose the cache. Every production router blends them, and the blend is
a tuning decision, not a solved problem.

??? question "If round-robin throws away the cache, why would anyone use it?"
    Because it is stateless and perfect at the one thing it does: it needs no
    coordination, no cache telemetry, and it balances load exactly. On traffic
    with little repetition, or with few replicas, the bookkeeping behind
    cache-aware routing can cost more than the cache saves. The point of this
    lecture is that on multi-turn chat the balance flips, which is exactly why
    the blend is a tuning decision rather than a solved problem.
    [Full answer](qa.md#if-round-robin-throws-away-the-cache-why-would-anyone-use-it)

Cheap approximations that work well:
- **Session affinity**: hash the conversation id, so the same conversation
  always lands on the same replica. Trivial, no coordination, and captures most
  of the win for chat.
- **Prefix-hash routing**: hash the first N tokens; **consistent hashing** (a
  scheme where adding or removing a replica moves only a slice of the keys
  instead of reshuffling everything) keeps it stable as replicas come and go.
- **Global KV store**: a shared cache tier so any replica can fetch a computed
  prefix. Kiely §5.3.3 (p.140) covers this; it's the G4 tier from §5.3.2.

---

## Part 2: Disaggregated prefill/decode

The bigger structural idea, and it follows directly from Lecture 01.

Prefill does the reading: it consumes the whole prompt at once, writes the KV
cache, and produces the first token. Decode does the talking: one token per
step, each depending on the last. Reading is compute-bound, talking is
memory-bound (Lecture 01). On one GPU they compete, and tuning for one hurts the
other. So **run them on different machines**:

```
request -> [PREFILL worker]  computes KV cache, first token
              |  transfer KV cache over the interconnect
              v
           [DECODE worker]   generates the rest
```

Each side can then be optimized independently, and that's the actual payoff:

| | Prefill workers | Decode workers |
|---|---|---|
| Bound by | compute | memory bandwidth |
| Want | high FLOPS | high bandwidth, big VRAM |
| Batch | small | large |
| Scale with | input tokens/sec | concurrent sequences |
| Parallelism | TP for latency | more replicas for throughput |

You can even use **different GPUs** for each, compute-dense cards for prefill,
bandwidth-dense ones for decode. (**TP** is tensor parallelism, splitting one
model across GPUs so a single request goes faster, Lecture 22.)

### The cost

**Transferring the KV cache.** The **interconnect** is the data link between
machines: PCIe or NVLink inside one server, InfiniBand or ethernet across a
rack, all far slower than the memory inside a single GPU. For a long prompt the
payload is huge. Lecture 09's arithmetic, unchanged:

```
32,768 tokens × 112 KiB per token  =  3,670,016 KiB  =  3.5 GiB
    128 tokens × 112 KiB per token  =     14,336 KiB  =  14 MiB
```

A long document's cache is gigabytes, big enough that the transfer takes real
time. A short chat's cache is 14 MiB, pocket change by size, yet it pays the
same fixed costs to cross the interconnect: a round-trip, a handshake, the
receiving worker's setup. Those fixed costs don't shrink with the payload,
which is why short prompts lose proportionally the most. Whether disaggregation
wins depends entirely on whether the transfer costs less than the interference
it removes.

**It wins when:** prompts are long, load is high, prefill and decode genuinely
contend, and you have fast interconnect.

**It loses when:** prompts are short (transfer dominates a cheap prefill), load is
light (no contention to remove), or interconnect is slow.

Kiely §5.5.2 (p.149) has the judgment call; note that it's a real judgment, not a
strict improvement. A **conditional** variant is common: send short prompts
straight to decode and disaggregate only the long ones.

---

## Build it

1. `serve/router.py`, multiple replicas, pluggable strategies: round-robin,
   session affinity, prefix-hash.
2. Measure **cache hit rate** for each on `shared_prefix` and on a simulated
   multi-turn conversation workload.
3. `serve/disaggregated.py`, separate prefill and decode workers with KV transfer
   between them.
4. Measure TTFT and TPOT for both architectures, on short and long prompts.
5. **Find the crossover**: at what prompt length does disaggregation start to win?

**Predict first:** guess the crossover length before measuring.

---

## What you should see

**Cache-aware routing much better hit rates** than round-robin on multi-turn
traffic, often the difference between a working prefix cache and a decorative one.

**Hotspots if you route purely by affinity.** Try it deliberately; the failure mode
is instructive.

**Disaggregation helping only on long prompts under load.** On short prompts the
transfer overhead makes it worse. If your measurement says it always wins, check
whether you're actually transferring the cache.

**More stable TPOT** when disaggregated, decode workers stop being interrupted.

---

## Go deeper

- **Kiely §5.3.3** (p.140), cache-aware routing, with the multi-replica diagram.
- **Kiely §5.5–5.5.3** (p.148–151), disaggregation, when to use it, and NVIDIA
  Dynamo's dynamic variant.
- **[DistServe](https://arxiv.org/abs/2401.09670)** (Zhong et al.), the
  disaggregation paper; §3 quantifies the interference you're removing.
- **[Splitwise](https://arxiv.org/abs/2311.18677)** (Patel et al.), same idea,
  with a strong argument for heterogeneous hardware per phase.
- **Gordić, *Inside vLLM***: the disaggregated P/D section, on how vLLM does the
  transfer.
- **[SGLang RadixAttention](https://arxiv.org/abs/2312.07104)**: prefix-aware
  routing with a radix tree.

---

## Check yourself

1. Why does round-robin routing waste your prefix cache specifically on multi-turn
   traffic?
2. What breaks if you route purely by cache affinity?
3. Why do prefill and decode workers want different hardware?
4. Under what conditions does disaggregation *lose*? Be specific.
5. From your crossover measurement: what routing policy would you actually deploy,
   and why?

---

## Next

**[28. Autoscaling and cost](28-autoscaling-and-cost.md)**: the number the whole
industry actually optimizes.

```bash
uv run python bench/cost_model.py
```

Run it before reading. Idling at 20% utilization costs **more than doubling your
batch size saves**: per token, 20% utilization is 4.97× pricier than 100%
(Lecture 28's worked example: $0.174/M vs $0.035/M), while doubling the batch
on a memory-bound phase is at most a 2× win (Lecture 01). The penalty dwarfs
the saving, which reframes most of Part III.

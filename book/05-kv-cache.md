# 05 — The KV cache

**Build:** `engine/cache.py::KVCache`, `engine/generate.py::generate_cached`
**Test:** `tests/test_05_kv_cache.py` · **Moves:** the curve goes flat; tok/s stops falling with position
**Prereq:** [04 — Measuring](04-measuring.md)

---

## The problem

You measured it in Lecture 03: per-token latency climbs with position, and the
demo counted the waste at **99.7%**. Generating 512 tokens computed 163,584 K/V
vectors to use 512 of them.

Now fix it.

---

## The idea

The fix follows from one property of causal attention:

> Token 5's key and value never depend on token 6.

Attention is causal — each token attends only to what came before. So when the
model computes `K` and `V` for token 5 at step 5, those tensors are **final**.
They don't change at step 6, or 400. Recomputing them is pure waste.

So don't. Store them.

That turns each step from "process the whole sequence" into "process one token,
attending against the stored past":

```
step n, no cache:   run n tokens through the model      O(n)
step n, with cache: run 1 token, attend over n cached   O(1) for K/V projection
```

Total generation cost drops from **O(N²)** to **O(N)** — for the projections. The
attention itself still reads all N cached entries per step, so decode remains
memory-bound (Lecture 02's 0.75 ops:byte). You haven't fixed the bottleneck; you
have stopped doing avoidable work on top of it.

### The two phases fall out naturally

This is where Lecture 01's split becomes structural rather than conceptual:

- **Prefill** — one forward pass over the whole prompt, filling the cache.
- **Decode** — one token at a time, appending one entry per step.

Same weights, two different code paths. Every engine in this book has this shape.

### What it costs

Nothing is free. From `recomputation.py`:

```
Storing K/V per token: 112 KiB
   seq len    cache size
       512         56 MiB
      8192        896 MiB
     32768       3584 MiB
```

You traded compute for memory, and the memory grows **linearly with context and
linearly with batch size**. That product is the central resource problem of LLM
serving:

```
cache_bytes = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes × seq_len × batch
```

For Qwen3-0.6B: 2 × 28 × 8 × 128 × 2 = **112 KiB per token**. At 32k context
that's 3.5 GiB for a *single* sequence — on a model whose weights are 1.2 GiB.

**The cache outgrows the model.** This is why Lectures 09 (paging) and 10 (prefix
sharing) exist: once you have a cache, the entire game becomes spending that
memory well.

> **GQA is doing real work here.** Qwen3-0.6B has 16 query heads but only 8 KV
> heads, and the cache is sized by *KV* heads. Grouped-query attention halves
> this number outright. Since decode is memory-bound, that's a direct 2× on the
> thing that bottlenecks you — which is why essentially every modern model uses
> it.

---

## The code

HuggingFace models already support caching; the mechanism is `use_cache` plus
threading `past_key_values` through the loop.

```python
def generate_cached(model, tokenizer, prompt, max_tokens=128, on_token=None):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # PREFILL: whole prompt, one pass. Builds the cache.
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1].argmax(-1, keepdim=True)

    generated = [next_id.item()]
    if on_token:
        on_token()

    # DECODE: one token in, one token out. Cache carries the past.
    for _ in range(max_tokens - 1):
        with torch.no_grad():
            out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values          # grown by one entry
        next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_id.item())
        if on_token:
            on_token()

    return tokenizer.decode(ids[0].tolist() + generated, skip_special_tokens=True)
```

The critical line is `model(next_id, past_key_values=past)` — **one** token goes
in, not the whole sequence. If you pass the full sequence *and* the cache, you get
wrong output and no speedup, which is the classic first bug here.

> **A note on the transformers API.** In transformers 5.x, `past_key_values` is a
> `Cache` **object**, not the legacy tuple-of-tuples you'll see in older tutorials.
> Treat it as opaque and thread it through — the code above works either way.
>
> Two implementations are worth knowing, because they're the same distinction
> you'll rebuild yourself:
>
> - **`DynamicCache`** — grows by concatenation as tokens arrive. The default,
>   and the direct analogue of the `KVCache` you're about to write.
> - **`StaticCache`** — pre-allocated to `max_cache_len`, fixed shape, written
>   in place. Wasteful for the same reason your `KVCache` will be (Lecture 09),
>   but the fixed shape is exactly what CUDA graphs require. That's why it
>   exists, and it comes back in Lecture 13.
>
> Notice that HuggingFace faced your Lecture 09 problem and shipped both answers.

### Then write it yourself

Using HuggingFace's cache teaches you the shape. Implementing `KVCache` teaches
you what's actually in it — and you need that for Lecture 09, where you'll replace
contiguous storage with blocks.

The data structure is a pre-allocated tensor per layer:

```python
class KVCache:
    def __init__(self, n_layers, max_seqs, max_seq_len, n_kv_heads, head_dim, ...):
        shape = (max_seqs, max_seq_len, n_kv_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.lengths = torch.zeros(max_seqs, dtype=torch.long)
```

Note what this costs: you must reserve `max_seq_len` for **every** sequence up
front, whether it uses 10 tokens or 10,000. A sequence that stops early holds its
full reservation until it's freed.

**Sit with that.** It's the flaw Lecture 09 fixes, and noticing it yourself now
makes paged attention feel inevitable rather than clever.

---

## Build it

1. Implement `generate_cached` in `engine/generate.py`.
2. `uv run pytest tests/test_05_kv_cache.py -v` — output must **exactly** match
   your `generate_naive` and HuggingFace. Same greedy path, same tokens.
3. Measure and overlay:

```bash
uv run python book/code/naive_bench.py --cached
```

4. Implement `KVCache` in `engine/cache.py` and compute its memory footprint for
   the batch/context combinations you'd want to serve.

**Record in `notes/01-engine/README.md`:** tok/s before and after at each length,
the speedup at 1024 tokens, and — most importantly — whether the *slope* went
flat, not just whether the number got bigger.

---

## What you should see

The climbing curve becomes roughly horizontal. Per-token time stops depending on
position.

Two honest caveats:

**The curve won't be perfectly flat.** Attention still reads the whole cache each
step, and that grows. You've removed the quadratic *projection* cost, not the
linear *attention* cost. On short sequences you may not see the residual slope at
all; on long ones you will.

**The speedup grows with length.** At 128 tokens it may be modest — fixed
overheads dominate. At 1024 it should be dramatic. If you only test short
sequences you'll under-measure the win, which is a nice illustration of why
workload choice determines what you can even see.

---

## Go deeper

- **Kiely §5.3** (p.136) — why every engine does this by default.
- **Kiely §5.3.2** (p.139) — the G1–G4 storage hierarchy: VRAM → host RAM →
  local SSD → networked. Foreshadows Lecture 27.
- **Kiely §5.4, Fig 5.11** (p.142) — the VRAM sizing formula. Apply it to a model
  you'd actually deploy; "will it fit" is the most common real question in this
  field.
- **[GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)**
  (Ainslie et al., 2023) — why 8 KV heads instead of 16.
- **[Field notes](field-notes.md)** — practitioners running 170k context on 2×3090.
  At that length the cache dwarfs the weights.

---

## Check yourself

1. Your speedup at 1024 tokens was larger than at 128. Why?
2. The cache made decode faster but its arithmetic intensity is *unchanged* at
   ~0.75 ops:byte. Explain why both are true.
3. Batch 32 sequences at 8k context with Qwen3-0.6B. How much KV cache? Now
   Llama-70B-scale (80 layers, 8 KV heads, 128 head dim). What breaks first?
4. `KVCache` reserves `max_seq_len` per sequence. A request that generates 10
   tokens with `max_seq_len=32768` — how much of its reservation is wasted?

That last one is Lecture 09.

---

**Next:** [06 — Sampling](06-sampling.md) — a short one, and it gives you the
determinism your tests depend on.

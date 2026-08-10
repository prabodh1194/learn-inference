# 10 — Prefix caching

**Build:** `BlockManager.match_prefix`, content hashing, LRU eviction
**Test:** `tests/test_10_prefix.py` · **Moves:** TTFT on shared-prefix traffic, often dramatically
**Prereq:** [09 — Paged attention](09-paged-attention.md)

---

## The problem

Look at what a real chat server actually receives:

```
request 1: [400-token system prompt] + "What is paged attention?"
request 2: [400-token system prompt] + "How does prefix caching work?"
request 3: [400-token system prompt] + "Explain continuous batching."
```

You prefill those 400 tokens every time. They're **identical** every time. The
keys and values are byte-for-byte the same, because attention is causal and those
tokens have the same predecessors in every request.

Agents, RAG, code assistants, and multi-turn chat all have this shape. In
production, shared prefixes are the norm rather than a special case.

---

## The idea

Lecture 09 gave you blocks. Blocks can be **shared**.

> If two sequences have identical tokens for a block's worth of positions, they
> can point at the **same physical block**. Compute it once, use it many times.

Requirements: a way to recognize "identical", and a way to know when it's safe to
free a block that several sequences point at.

### Content hashing

Identify blocks by what's *in* them, not by who allocated them. The subtlety is
that a block's contents aren't determined by its own tokens alone:

```python
block_hash = hash((parent_hash, tuple(token_ids_in_this_block)))
```

The **parent hash is essential.** Keys and values at position 400 depend on
tokens 0–399. Two blocks with identical tokens but different histories produce
different K/V and must not be shared. Chaining the parent hash into each block's
identity encodes "same tokens *and* same history."

Get this wrong and you produce subtly wrong output on cache hits, which is
brutal to debug, because it only manifests under specific traffic.

### Reference counting

You built the hook in Lecture 09. Now it earns its keep:

```
block 42 (system prompt, tokens 0-15)   ref_count = 3
```

Three sequences use it. When one finishes, decrement to 2, **don't free.** Only
at zero does the block become reclaimable.

### Eviction

A block at refcount 0 isn't garbage; it's *cached*. Keep it around in case
another request wants the same prefix, and only reuse the physical block when you
need memory.

That makes it an LRU cache over refcount-zero blocks:

```
allocate:  free list empty?
             -> evict the least-recently-used refcount-0 block
             -> if none exist, we are genuinely out of memory (preempt)
```

### The rule that determines your savings

Here is the part that surprises people, and it's worth internalizing precisely:

> **A prefix ends at the first differing token.** Everything after it is a miss,
> even if identical.

Consider two prompts with the *same tokens in a different order*:

```
A: "Weather in SF?"     vs  "Weather in NYC?"     -> shares "Weather in"
B: "SF weather today?"  vs  "NYC weather today?"  -> shares NOTHING
```

In case B every token after the first differs *positionally*, so nothing is
reusable, despite the prompts being nearly identical to a human reader.

**Practical consequence:** put stable content first and novel content last. System
prompt, then retrieved documents, then conversation history, then the user's new
message. Reorder those and you can lose the entire cache benefit while changing
nothing a user would notice.

This is why pay-per-token APIs bill "cache hits" cheaper, and why prompt
*ordering* is a real engineering lever, not a style preference.

`bench/workloads.py` ships `shared_prefix` and `late_divergence` precisely to make
this measurable: near-identical token counts, wildly different hit rates.

---

## The code

```python
def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
    """Find cached blocks matching this token sequence's prefix.

    Returns (block_ids, n_tokens_hit). Stops at the first miss -- once the
    chain breaks, nothing downstream can match.
    """
    matched, parent_hash = [], None
    for i in range(0, len(token_ids), self.block_size):
        chunk = token_ids[i:i + self.block_size]
        if len(chunk) < self.block_size:
            break                       # only full blocks are cacheable

        h = hash((parent_hash, tuple(chunk)))
        if h not in self.hash_to_block:
            break                       # first miss ends the prefix
        block = self.hash_to_block[h]
        self.ref_counts[block] += 1     # we now hold a reference
        self.lru.touch(block)
        matched.append(block)
        parent_hash = h

    return matched, len(matched) * self.block_size
```

**Only full blocks are cacheable.** A partially-filled block is still being
written; hashing it would let a later, longer sequence match a prefix whose
contents haven't been finalized.

**Stop at the first miss.** Once the hash chain breaks, every later block's parent
hash differs too. Continuing is wasted work.

Then in the scheduler:

```python
cached_blocks, n_hit = block_manager.match_prefix(seq.prompt_ids)
seq.block_table = cached_blocks
seq.num_prefilled = n_hit          # skip prefill for these tokens entirely
```

Those `n_hit` tokens never touch the GPU. That's the whole win.

---

## Build it

1. Add content hashing with parent chaining to `BlockManager`.
2. Implement `match_prefix`, refcount increments on hit, and LRU eviction over
   refcount-0 blocks.
3. Wire it into admission, set `num_prefilled` so prefill skips the hit region.
4. `uv run pytest tests/test_10_prefix.py -v`, **cache hits must not change
   output.** Same tokens either way.
5. Measure the contrast that teaches the lesson:

```bash
uv run python book/code/prefix_bench.py
```

This runs `shared_prefix` and `late_divergence` back to back. **Predict the TTFT
for each before running.**

---

## What you should see

**TTFT collapses on `shared_prefix`** after the first request. Those 400 tokens
are already computed; you skip nearly all prefill.

**Essentially nothing on `late_divergence`** — same token count, first token
differs, zero reuse.

**Throughput up too**, since skipped prefill frees the GPU for decode.

That side-by-side is the lecture. Two workloads, nearly identical cost on paper,
completely different results.

---

## Go deeper

- **[Efficient Memory Management ... PagedAttention](https://arxiv.org/abs/2309.06180)**
  §4.3, copy-on-write and sharing, which you just built.
- **[SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)** (Zheng et al.) —
  a radix tree instead of a flat hash map, so prefixes share *structurally*.
  Better for branching conversation trees.
- **Kiely §5.3.1** (p.136–138): the ordering rule, with the SF/NYC example this
  lecture borrows.
- **vLLM `vllm/v1/core/kv_cache_utils.py`** — production block hashing. Note how
  much care goes into what's included in the hash (LoRA id, multimodal inputs) —
  every one of those is a correctness bug someone hit.

---

## Check yourself

1. Why must the parent hash be part of a block's identity? Construct a case that
   breaks without it.
2. `"Weather in SF?"` / `"Weather in NYC?"` share a prefix; `"SF weather today?"`
   / `"NYC weather today?"` share nothing. Same words. Explain.
3. Why are partially-filled blocks not cacheable?
4. A refcount-0 block is kept rather than freed. Why, and when is it reclaimed?
5. You have a 2000-token system prompt and 50-token user messages. Estimate the
   TTFT saving. What would you tell an application developer about prompt
   ordering?

---

## Next

**[11 — Chunked prefill](11-chunked-prefill.md)** — fixing the p99 damage L08
caused.

```bash
uv run python book/code/chunked_bench.py
```

**Judge this one on p99, not the mean.** The mean barely moves, if that's all
you watch, you'll conclude the lecture did nothing.

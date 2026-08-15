# 10. Prefix caching

**Build:** `BlockManager.match_prefix`, content hashing, LRU eviction
**Test:** `tests/test_10_prefix.py` · **Moves:** TTFT on shared-prefix traffic, often dramatically
**Prereq:** [09. Paged attention](09-paged-attention.md)

---

## The problem

Look at what a real chat server actually receives:

```
request 1: [400-token system prompt] + "What is paged attention?"
request 2: [400-token system prompt] + "How does prefix caching work?"
request 3: [400-token system prompt] + "Explain continuous batching."
```

You prefill those 400 tokens every time. They're **identical** every time, and
identical input produces identical scratchpad entries. Recall the KV cache from
Lecture 05: while the model reads a prompt it writes a record of every token
into memory, two entries per token per layer (K and V), the numbers that later
tokens will attend to. Each token's record is computed from the tokens that
came before it, and attention is **causal**, each position only ever sees the
positions to its left, never to its right (Lecture 05). Position 0's record
depends on nothing but itself, position 1's on tokens 0–1, position 399's on
tokens 0–399, always the same chain:

```
request 1:  tok 0 → tok 1 → tok 2 → ... → tok 399 → "What is paged attention?"
request 2:  tok 0 → tok 1 → tok 2 → ... → tok 399 → "How does prefix caching work?"
request 3:  tok 0 → tok 1 → tok 2 → ... → tok 399 → "Explain continuous batching."

the first 400 links are one and the same chain, so every request computes
the same 400 records, three times over.
```

The keys and values are byte-for-byte the same in all three, and you recompute
them every time, 400 tokens of arithmetic per request, for results you already
hold.

Agents, RAG, code assistants, and multi-turn chat all have this shape. In
production, shared prefixes are the norm rather than a special case.

---

## The idea

Lecture 09 gave you blocks. Blocks can be **shared**.

> If two sequences have identical tokens for a block's worth of positions, they
> can point at the **same physical block**. Compute it once, use it many times.

```
sequence 1:  [block 4: system prompt] [block 7: user message]
sequence 2:  [block 4: system prompt] [block 9: user question]
                    └─ one physical block, two pointers, computed once ─┘
```

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
different K/V and must not be shared. Concretely: suppose the 16 tokens
`[11, 22, 33, 44]` appear at the *start* of one prompt and at position 2,000 of
another. The tokens are the same, but the first block's records were computed
with nothing before them, the second's with 2,000 tokens of context: different
numbers, byte for byte. A hash of the block's own tokens alone would call them
identical and serve the wrong K/V. Chaining the parent hash into each block's
identity encodes "same tokens *and* same history."

??? question "How can two blocks hold identical token ids but different histories?"
    The same 16 token ids can appear at different *positions* in two prompts.
    A token's K/V record is computed from everything the model had seen before
    that token, so the record at position 2,000 includes 2,000 tokens of
    context the record at position 0 never had. Same tokens, different numbers.
    The parent hash exists to tell these two cases apart.
    [Full answer](qa.md#how-can-two-blocks-hold-identical-token-ids-but-different-histories)

Get this wrong and you produce subtly wrong output on cache hits, which is
brutal to debug, because it only manifests under specific traffic.

### Reference counting

You built the hook in Lecture 09. Now it earns its keep:

```
block 42 (system prompt, tokens 0-15)   ref_count = 3
```

Three sequences use it. When one finishes, decrement to 2, **don't free.** Only
at zero does the block become reclaimable.

### Copy-on-write

Sharing so far is read-only: everyone points at the same frozen prefix. But a
sequence must *write* K/V for every token it generates. What happens when a
shared block's owner needs to write into it?

```
block 42 (system prompt, tokens 0-15)      ref_count = 3
sequence 3's next token needs its K/V written into block 42
```

Writing in place would corrupt sequences 1 and 2. The rule: **a block with
refcount > 1 is never written in place.** The writer copies the block, writes
into the private copy, and decrements the original's count:

```
copy block 42 -> block 51, ref_count(51) = 1, sequence 3 now points at 51
                 ref_count(42) = 2, sequences 1 and 2 still share it
```

One block copied, not one sequence. That is **copy-on-write** — the same trick
an OS uses for `fork()` — and it is what makes sharing work for sequences that
are about to *diverge*, not just sequences that never will.

**Parallel sampling.** A prompt sampled N times (Lecture 06) runs N sequences
that share the prompt's blocks. They only differ in generated tokens, and a
sequence copies a block only at the moment it must write into a shared one:
until then, one physical copy serves all N.

**Beam search.** Instead of one continuation, keep the K most promising
candidate continuations alive and re-rank them each step. The candidates share
their generated history too: when a beam diverges it copies exactly one block;
when a beam is pruned its blocks are decremented, and only the last beam to
let go of a block frees it. The paper reports up to 55% of beam search's KV
memory saved this way. In a contiguous world every branch and every prune is a
full-sequence KV copy; with paging it is one block copy or one decrement.

> Notice nothing new was needed. The block table and the refcount were built
> for prefixes; in-request sharing (samples, beams) and cross-request sharing
> (prefixes) are the same two mechanisms at work.

??? question "Why can't a sequence just write into a block it shares?"
    Because K/V are written in place: if sequence 3 wrote into block 42, it
    would silently corrupt the K/V that sequences 1 and 2 still attend to.
    The fix is copy-on-write: refcount > 1 means write into a fresh copy and
    decrement the original. The copy is one block of 16 tokens, never the
    whole sequence.
    [Full answer](qa.md#why-cant-a-sequence-just-write-into-a-block-it-shares)

### Eviction

A block at refcount 0 isn't garbage; it's *cached*. Keep it around in case
another request wants the same prefix, and only reuse the physical block when you
need memory.

That makes it an LRU cache (least-recently-used: when you must give up memory,
drop the block that has gone longest without being wanted) over refcount-zero
blocks:

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

**Essentially nothing on `late_divergence`**: same token count, first token
differs, zero reuse.

**Throughput up too**, since skipped prefill frees the GPU for decode.

That side-by-side is the lecture. Two workloads, nearly identical cost on paper,
completely different results.

---

## Go deeper

- **[Efficient Memory Management ... PagedAttention](https://arxiv.org/abs/2309.06180)**
  §4.3, copy-on-write and sharing, which you just built.
- **[SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)** (Zheng et al.),   a radix tree instead of a flat hash map, so prefixes share *structurally*.
  Better for branching conversation trees.
- **Kiely §5.3.1** (p.136–138): the ordering rule, with the SF/NYC example this
  lecture borrows.
- **vLLM `vllm/v1/core/kv_cache_utils.py`**: production block hashing. Note how
  much care goes into what's included in the hash (LoRA id, multimodal inputs),
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

**[11. Chunked prefill](11-chunked-prefill.md)**: fixing the p99 damage L08
caused.

```bash
uv run python book/code/chunked_bench.py
```

**Judge this one on p99, not the mean.** The mean barely moves, if that's all
you watch, you'll conclude the lecture did nothing.

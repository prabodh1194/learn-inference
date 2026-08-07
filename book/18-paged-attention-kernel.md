# 18 — A paged attention kernel

**Build:** `kernels/triton/paged_attention.py` · **Test:** `tests/test_18_paged_kernel.py` (cuda)
**Moves:** decode attention latency — recovers what Lecture 09 cost you
**Prereq:** [17 — FlashAttention](17-flash-attention.md), [09 — Paged attention](09-paged-attention.md)

---

## The problem

Lecture 09 bought you a large increase in concurrent sequences, and charged you
for it. Your PyTorch gather:

```python
gathered = cache[blocks]                        # materialize every block
flat = gathered.reshape(-1, *cache.shape[2:])   # copy again
return flat[:seq_len]                           # and trim
```

That reassembles the entire logical K/V sequence in HBM before attention even
starts — reintroducing exactly the round-trip Lecture 17 just eliminated. You
have FlashAttention and paging, and they're fighting each other.

The fix is to teach the kernel to read block tables directly.

---

## The idea

FlashAttention already loads K/V **in tiles**. Paging stores K/V **in blocks**.
These are the same shape of thing.

> Make the tile loop iterate over *blocks via the block table* instead of over
> contiguous positions.

Nothing else about the algorithm changes. Same online softmax, same running max
and accumulator, same rescaling. Only the address computation differs:

```python
# FlashAttention: contiguous
k_tile = tl.load(K + start_n * stride + offsets)

# PagedAttention: indirect
block_id = tl.load(block_table_ptr + block_idx)      # where does this tile live?
k_tile = tl.load(K_cache + block_id * block_stride + offsets)
```

One extra load — the block table lookup — per tile. The block table is tiny and
stays in cache, so the overhead is small. The gather disappears entirely.

### Decode is the special case

Prefill attends with many queries. **Decode has exactly one query token** attending
over the whole cached context. That changes the shape of the problem:

- No query tiling — one row.
- No causal masking within the tile — the single query attends to everything
  cached, all of which precedes it.
- The whole kernel is dominated by **streaming K/V from memory**, which is
  Lecture 02's memory-bound decode, in kernel form.

Because it's pure bandwidth, the metric from Lecture 15 is the right one: measure
achieved bandwidth against peak. A good decode attention kernel gets close.

### Parallelizing over context

With one query, a naive kernel launches one block and leaves the GPU nearly idle
at small batch sizes. The standard fix is to **split the context across blocks**,
each computing a partial (max, sum, accumulator), then combine:

```
block 0: tokens    0-1023  -> (m_0, l_0, acc_0)
block 1: tokens 1024-2047  -> (m_1, l_1, acc_1)
                              ↓ combine with the same rescaling rule
                          final output
```

The combination uses the *identical* online-softmax merge from Lecture 17 —
rescale each partial by `exp(m_i - m_global)`, sum, divide. Having built that
already, this is a small step rather than a new idea.

This is what vLLM calls the "split-KV" or FlashDecoding path, and it's why decode
attention scales down to batch 1 without wasting the GPU.

---

## Build it

1. Start from your Lecture 17 kernel. Change **only** the K/V addressing to go
   through the block table.
2. `uv run pytest tests/test_18_paged_kernel.py -v` — must match both your
   contiguous FlashAttention **and** the Part II PyTorch path. Three
   implementations agreeing is strong evidence.
3. Handle the **partial last block**: a sequence of 37 tokens with `block_size=16`
   has 5 valid tokens in its third block. Mask the rest to `-inf` before softmax,
   or they contribute garbage weights.
4. Add context-splitting for batch-1 decode.
5. Benchmark against the PyTorch gather at several context lengths, and re-run
   end-to-end.

---

## What you should see

**Large speedup versus the gather**, growing with context length — you removed a
copy proportional to sequence length.

**Most of Lecture 09's per-step cost recovered.** Paging becomes close to free,
which is the whole point: keep the memory win, drop the latency penalty.

**Bandwidth-bound decode attention.** If you're near peak, you're done; the
remaining gap is in the linear layers.

---

## Go deeper

- **[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)** §4 — re-read now
  that you've written the kernel; the memory-manager design reads differently.
- **[FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)** —
  the context-splitting idea, explained well and short.
- **vLLM `vllm/v1/attention/backends/triton_attn.py`** — the production Triton
  path. Compare its block-table handling to yours.
- **vLLM `csrc/attention/paged_attention_v1.cu`** — the CUDA version, if you want
  a preview of Lecture 20.

---

## Check yourself

1. Why is a block table lookup per tile cheap, when the PyTorch gather was
   expensive?
2. What's different about decode attention versus prefill attention, and why does
   it change the kernel?
3. A sequence has 37 tokens with `block_size=16`. What must the kernel do with the
   third block?
4. Why does splitting the context across thread blocks help at batch 1 but not at
   batch 64?
5. Compare per-step latency now against Lecture 09's PyTorch version. How much of
   the paging cost did you recover?

---

**Next:** [19 — Quantization](19-quantization.md) — make the bytes smaller, and
learn to measure what it costs you.

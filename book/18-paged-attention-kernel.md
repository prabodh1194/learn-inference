# 18. A paged attention kernel

**Build:** `kernels/triton/paged_attention.py` · **Test:** `tests/test_18_paged_kernel.py` (cuda)
**Moves:** decode attention latency, recovers what Lecture 09 cost you
**Prereq:** [17. FlashAttention](17-flash-attention.md), [09. Paged attention](09-paged-attention.md)

---


## The problem

Lecture 09 scattered the KV cache into blocks. Lecture 17 wrote attention
assuming one contiguous span. Both are wins; together they fight, so your engine
currently reconciles them the expensive way — it un-scatters the cache before
every attention call:

```python
gathered = cache[blocks]                        # copy every block
flat = gathered.reshape(-1, *cache.shape[2:])   # copy again
return flat[:seq_len]
```

That copies every cached byte into a temporary strip, reads it once, and throws
it away — reintroducing exactly the HBM round-trip Lecture 17 existed to remove.

---

## The idea

Teach the kernel to read the block table itself. The read changes from address
arithmetic to a table lookup:

```python
# before: the next chunk is where arithmetic says it is
k = tl.load(K + start_n * stride + offsets)

# after: ask where it lives, then read there
block_id = tl.load(block_table_ptr + block_idx)
k = tl.load(K_cache + block_id * block_stride + offsets)
```

Same online softmax, same running max, same accumulator. Only the address
computation moves.

If you write systems code this is unremarkable — it is a page table. It is also
cheap for a reason worth one line: the index is 4 bytes and the block it finds is
64 KiB (`16 tokens × 4,096 B` per layer, from Lecture 05), so you move address
bytes instead of payload bytes. A 32k-token sequence's whole table is 8 KiB.

**The substitution is the easy part.** Three things about it are not, and they
are the rest of the lecture.

---

## 1. A tile is several blocks

The kernel and the allocator picked different chunk sizes, for different
reasons, and neither yields:

```
   tile    BLOCK_N ≈ 64–128 tokens    kernel's choice, sized to fill SRAM
   block   block_size = 16 tokens     allocator's choice, sized to limit waste
```

So one tile is four blocks, at four unrelated addresses. "One load" is really a
gather, assembled on chip:

```
   one tile of 64 tokens = 4 blocks
   ┌────────┬────────┬────────┬────────┐
   │ blk 93 │ blk 12 │ blk 57 │ blk  8 │   4 lookups, 4 scattered reads
   └────────┴────────┴────────┴────────┘
```

The only constraint: a tile must be a whole number of blocks, so no tile
boundary lands mid-block.

## 2. The tile no longer fits

Lecture 17 chose `BLOCK_N = 64` because K and V tiles fit in ~100 KB of SRAM. At
4,096 B per token per layer, they no longer do:

```
   K  64 × 2,048 B = 128 KiB
   V  64 × 2,048 B = 128 KiB
                     ────────
                     256 KiB   against ~100 KB
```

Paged decode kernels answer with a smaller `BLOCK_N`, or by streaming blocks
through rather than holding a whole tile. Decode can afford this because its
query side is a single row — `Br = 1`, so the score patch is `1 × BLOCK_N`
instead of `64 × 64`, and almost the whole scratchpad is free for K/V.

!!! warning "Use the per-layer K/V figure"
    `4,096 B` is one token's K/V in **one layer**. Lecture 05's `114,688 B` is
    that × 28 layers — the whole-model figure, right for cache capacity, wrong
    here by 28×. If a tile looks absurdly large for SRAM, suspect this first.

## 3. Split-KV — the only real algorithm change

Everything above is bookkeeping. This is not: it changes how the work is
divided, and needs a merge step with no counterpart in Lecture 17.

Decode gives one query row per sequence, and attention parallelizes over
(sequence, head). At batch 1 with 16 query heads that is 16 thread blocks — on a
3090 with **82 SMs**:

```
   16 blocks / 82 SMs  ≈  20% occupancy      66 SMs idle
```

No addressing fix helps; there simply is not enough work. So manufacture some:
split the context across blocks, let each compute a partial `(m, l, acc)`, then
merge.

```
   tokens 0-1023 ──► block 0 ──► (m₀, l₀, acc₀) ─┐
                                                  ├─► rescale, add, divide
   tokens 1024+  ──► block 1 ──► (m₁, l₁, acc₁) ─┘
```

The merge is Lecture 17's online softmax, one level up: rescale each partial by
`exp(mᵢ − m_global)`, sum, divide. Check it on four tokens, `d = 2`, one query,
split in two — scores `[1, 3, 2, 0]`:

```
   one-shot:   m = 3    l = 1.5530    out = [0.3881, 0.9449]

   block 0  scores [1,3]   m₀ = 3   l₀ = 1.1353   acc₀ = [0.1353, 1.0000]
   block 1  scores [2,0]   m₁ = 2   l₁ = 1.1353   acc₁ = [1.2707, 1.2707]

   m_global = 3
   block 0 × exp(3−3) = 1.0000      it held the max; untouched
   block 1 × exp(2−3) = 0.3679      its max was too small; shrink

   l   = 1.5530                                    ✓
   out = [0.6028, 1.4675] / 1.5530 = [0.3881, 0.9449]   ✓ identical
```

No block knows anything about the others while it runs, and the merge needs only
three numbers from each — not the scores. That is what makes it parallel. vLLM
calls this split-KV, or FlashDecoding.

**One masking note.** Decode needs no *causal* mask — the single query attends to
everything cached. It still needs a *validity* mask: the last block is usually
partly empty (37 tokens fills two blocks and 5 slots of a third), and those
unwritten slots must go to `-inf` or they contribute real softmax weight.
Causal masking hides the future; this hides the unwritten.

---

## Build it

1. Start from your Lecture 17 kernel. Change **only** the K/V addressing to go
   through the block table.
2. `uv run pytest tests/test_18_paged_kernel.py -v`, must match both your
   contiguous FlashAttention **and** the Part II PyTorch path. Three
   implementations agreeing is strong evidence.
3. Handle the **partial last block**: a sequence of 37 tokens with `block_size=16`
   has `37 − 2×16 = 5` valid tokens in its third block. Mask the rest to `-inf`
   before softmax, or they contribute garbage weights.
4. Add context-splitting for batch-1 decode.
5. Benchmark against the PyTorch gather at several context lengths, and re-run
   end-to-end.

---

## What you should see

**Large speedup versus the gather**, growing with context length, you removed a
copy proportional to sequence length.

**Most of Lecture 09's per-step cost recovered.** Paging becomes close to free,
which is the whole point: keep the memory win, drop the latency penalty.

**Bandwidth-bound decode attention.** If you're near peak, you're done; the
remaining gap is in the linear layers.

---

## Go deeper

- **[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)** §4, re-read now
  that you've written the kernel; the memory-manager design reads differently.
- **["What is vLLM? | PagedAttention | Fully Explained: an OS Trick for 4× Throughput"](https://www.youtube.com/watch?v=xgl9Qrz31Mc)**
  (Papers by Hand): the kernel-engineering side of paging — the fused
  block-table lookup, the write-side kernels that place freshly computed K/V
  straight into scattered blocks, and why block size 16 is a hardware choice
  as much as a memory one.
- **[FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)**:   the context-splitting idea, explained well and short.
- **vLLM `vllm/v1/attention/backends/triton_attn.py`**: the production Triton
  path. Compare its block-table handling to yours. Note that V1 moved paged
  attention here from hand-written CUDA; the old
  `csrc/attention/paged_attention_v1.cu` no longer exists on `main`, and lives
  only in pre-V1 tags.

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

## Next

**[19. Quantization](19-quantization.md)**: make the bytes smaller, and learn
to measure what it costs you.

> **Build the quality harness BEFORE you benchmark speed.** This is the only
> optimization in the book that can make your model *worse*, and the damage is
> invisible unless you go looking for it. That ordering is the lecture.

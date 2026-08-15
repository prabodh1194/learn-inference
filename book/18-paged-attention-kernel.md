# 18. A paged attention kernel

**Build:** `kernels/triton/paged_attention.py` · **Test:** `tests/test_18_paged_kernel.py` (cuda)
**Moves:** decode attention latency, recovers what Lecture 09 cost you
**Prereq:** [17. FlashAttention](17-flash-attention.md), [09. Paged attention](09-paged-attention.md)

---

## The problem

Lecture 09 bought you a large increase in concurrent sequences, and charged you
for it. To see the charge, recall the memory layout it left you with. A
sequence's K/V no longer sits in one contiguous strip; it is scattered in
fixed-size **blocks**, and a **block table** (a short list, one per sequence
in flight) records which physical block holds which chunk. That scattered layout
is exactly what let you pack far more concurrent sequences into the same VRAM.

The catch: attention, as Lecture 17 wrote it, expects K/V as one contiguous
span. So every single decode step your engine has to undo the paging before
attention can run. That undo is your PyTorch gather, the three lines that
reassemble a sequence from its scattered blocks:

```python
gathered = cache[blocks]                        # materialize every block
flat = gathered.reshape(-1, *cache.shape[2:])   # copy again
return flat[:seq_len]                           # and trim
```

Concretely it copies every K and V value out of its scattered block, glues them
into one long strip, and hands that strip to attention. Draw it for a sequence
of three blocks:

```
logical sequence 0..37        physical blocks (scattered in VRAM)
[chunk 0][chunk 1][chunk 2]    block table: [4, 1, 9]
    ┆       ┆       ┆              ┆    ┆    ┆
    └───────┴───────┴──────────────┘    ┆    ┆   (copy chunk 0, from block 4)
                                     └───┴────┘   (copy chunks 1, 2 from 1, 9)
                                          ↓
                            one big contiguous strip in HBM
                                          ↓
                            attention reads the whole strip
```

`cache[blocks]` **materializes**: it copies the bytes out of their scattered
blocks into a brand-new contiguous strip. `reshape` and the slice handle that
strip again. Attention then reads it once more. Every byte of the
whole cached sequence is touched multiple times, every step, for every sequence
in the batch, and the strip you built was only ever a staging area to throw
away.

That is the **round-trip** Lecture 17 spent all its effort eliminating: bytes
go out to the big slow memory (**HBM**, the GPU's main RAM, the memory this
lecture's on-chip SRAM scratchpad is ~20× faster than) and come back again. Paging
scattered the data to save memory; FlashAttention assumed contiguous data to
save traffic. You have both techniques now, working at cross purposes. That's
the fight the lecture opener promised.

The fix is to teach the kernel to read block tables directly, so the scatter
never has to be undone.

---

## The idea

Two words to settle first, because they name the same thing from two lectures.
FlashAttention loads K/V **in tiles**: a chunk of the sequence, a few thousand
tokens wide, that the kernel pulls into fast on-chip memory, works on, and
never writes back (Lecture 17). Paging stores K/V **in blocks**: fixed-size
chunks, 16 tokens by default, held in scattered locations (Lecture 09). A tile
and a block are the same shape of thing, a chunk of K/V handled differently.
That overlap is the whole trick of this lecture.

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

One extra load (the block table lookup) per tile. The block table is tiny and
stays in fast on-chip memory (**SRAM**), so the overhead is small. The gather
disappears entirely.

Note the asymmetry that makes the lookup cheap and the gather expensive. The
gather moved *every byte* of the cached sequence: bytes out of their blocks,
bytes into a duplicate strip, and attention read the duplicate. The block table
lookup moves a few bytes: the address of the next tile's home. That one
address steers a much larger load. With `block_size = 16` and each token's K/V
costing 114,688 bytes (from Lecture 05), one tile holds:

```
16 tokens × 114,688 B/token  =  1,835,008 B  ≈  1.75 MiB
```

And the lookup that finds it reads a single 32-bit index, 4 bytes. A 4-byte
read steering a ~1.75 MiB read is why the indirection costs almost nothing
while the gather it replaced cost a full extra copy of the sequence (224 MiB
at the 2048-token context Lecture 02 used). You're spending 4 bytes to save
megabytes, every tile.

The write side gets the same treatment. Every step the model computes fresh K/V
for the batch's newest tokens, and a naive engine would stage them in a
contiguous scratch buffer, then copy them into blocks. vLLM fuses the placement
instead: a kernel that reshapes the freshly computed K/V and writes it straight
into each sequence's assigned physical block — one scatter write, no staging
copy. Same philosophy as the read side: bytes never make the round trip
through a contiguous staging area.

### Decode is the special case

Prefill attends with many queries, one per prompt token, side by side.
**Decode has exactly one query token** (the single new token, asking attention
to look back over everything cached) attending over the whole cached context.
That changes the shape of the problem:

- No query tiling, one row.
- No causal masking within the tile: the single query attends to everything
  cached, all of which precedes it.
- The whole kernel is dominated by **streaming K/V from memory**: reading each
  byte exactly once, on its way through, never holding it. That is Lecture
  02's memory-bound decode, in kernel form.

Because it's pure bandwidth, the metric from Lecture 15 is the right one: measure
achieved bandwidth against peak. A good decode attention kernel gets close.
Bandwidth here means bytes per second memory can hand over; on a
memory-bound kernel, achieved bandwidth close to peak is what "done" looks
like.

### Parallelizing over context

With one query, a naive kernel launches one block and leaves the GPU nearly idle
at small batch sizes. The standard fix is to **split the context across blocks**,
each computing a partial (max, sum, accumulator), then combine:

??? question "Wait: 'block' here is a thread block, not the KV block from Lecture 09?"
    Both words are in play, and they mean different things. The **KV block** is
    a fixed-size chunk of cached data, 16 tokens, something you allocate and
    free. The **thread block** is a group of threads that run together on one
    SM (one of the chip's work groups, each with its own fast private memory),
    a unit of execution. Splitting the context across thread blocks means
    different groups of threads each handle a slice of the sequence. Same word,
    unrelated meanings; the sentence around it tells you which is meant.
    [Full answer](qa.md#wait-block-here-is-a-thread-block-not-the-kv-block-from-lecture-09)

```
block 0: tokens    0-1023  -> (m_0, l_0, acc_0)
block 1: tokens 1024-2047  -> (m_1, l_1, acc_1)
                              ↓ combine with the same rescaling rule
                          final output
```

The combination uses the *identical* online-softmax merge from Lecture 17,
rescale each partial by `exp(m_i - m_global)`, sum, divide. Having built that
already, this is a small step rather than a new idea. In words: each block ran
softmax with its own local max `m_i`; the block with the largest max, `m_global`,
had it right, but the others used a max they later learned was too small, so
their accumulators are inflated by `exp(m_i - m_global)`. Shrink each partial
back to what a global max would have produced, add the pieces, divide by the
corrected sum: one attention output, assembled from independent slices.

This is what vLLM calls the "split-KV" or FlashDecoding path, and it's why decode
attention scales down to batch 1 without wasting the GPU.

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

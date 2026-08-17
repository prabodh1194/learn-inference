# 18. A paged attention kernel

**Build:** `kernels/triton/paged_attention.py` · **Test:** `tests/test_18_paged_kernel.py` (cuda)
**Moves:** decode attention latency, recovers what Lecture 09 cost you
**Prereq:** [17. FlashAttention](17-flash-attention.md), [09. Paged attention](09-paged-attention.md)

---

## The problem

You finished the last two lectures holding two techniques that contradict each
other:

```
   Lecture 09  paged the KV cache into scattered blocks   →  ~14× more sequences
   Lecture 17  wrote attention assuming ONE contiguous span →  O(N) traffic
```

Both are wins. Together they fight, because attention cannot read a block table.
So every decode step your engine currently undoes the paging first — a gather
that copies the scattered blocks back into one strip:

```python
gathered = cache[blocks]                        # materialize every block
flat = gathered.reshape(-1, *cache.shape[2:])   # copy again
return flat[:seq_len]                           # and trim
```

`cache[blocks]` **materializes**: it copies every cached byte into a brand-new
contiguous strip, attention reads that strip, and the strip is thrown away at
the end of the step.

```
   blocks 4, 1, 9  ──copy──►  [ chunk 0 │ chunk 1 │ chunk 2 ]  ──►  attention
   (scattered in VRAM)         a temporary strip in HBM             reads it

   every K/V byte: read out, written back, read again — every step,
   every sequence, and the strip is discarded immediately
```

That is exactly the **round-trip** Lecture 17 spent its entire effort
eliminating, reintroduced by the allocator. Paging scattered the data to save
memory; FlashAttention assumed contiguous data to save traffic.

The fix is one idea: **teach the kernel to read the block table itself**, so the
scatter never has to be undone.

---

## The idea

Two words name similar things from two lectures, and getting their relationship
right is the whole trick:

```
   tile   (L17)  a chunk of K/V the kernel LOADS at once    unit of COMPUTE
                 sized to fill SRAM        →  BLOCK_N, typically 64–128 tokens

   block  (L09)  a chunk of K/V the allocator OWNS          unit of STORAGE
                 sized to limit waste      →  block_size,  typically 16 tokens
```

**They are not the same size, and they are not chosen by the same person.** The
allocator picked 16 to bound internal fragmentation (L09); the kernel picks 64
or 128 so a tile fits in SRAM alongside Q and the score patch (L17). Neither
gets to overrule the other.

So the requirement is not that they match. It is only that **a tile is a whole
number of blocks**, so a tile boundary never lands in the middle of a block:

```
   one tile, BLOCK_N = 64 tokens
   ┌───────────────────────────────────────────────────┐
   │ block  │ block  │ block  │ block  │                │   4 lookups,
   │ 16 tok │ 16 tok │ 16 tok │ 16 tok │                │   4 possibly-distant
   └────────┴────────┴────────┴────────┘                    physical locations
       ↑        ↑        ↑        ↑
     tbl[0]   tbl[1]   tbl[2]   tbl[3]     ← block table entries for this tile
```

Get that right and the fight ends: the kernel keeps the tile size it wants, the
allocator keeps the block size it wants, and the block table translates between
them.

> Make the tile loop gather its blocks *through the block table* instead of
> reading one contiguous span.

Nothing else about the algorithm changes. Same online softmax, same running max
and accumulator, same rescaling. Only the address computation differs:

```python
# FlashAttention: one contiguous span, address by arithmetic
k_tile = tl.load(K + start_n * stride + offsets)

# PagedAttention: BLOCK_N/block_size blocks, each found by lookup
for b in range(BLOCK_N // block_size):               # e.g. 64/16 = 4
    block_id = tl.load(block_table_ptr + block_idx + b)
    k_part   = tl.load(K_cache + block_id * block_stride + offsets)
    # ...assembled into the tile, in registers/SRAM — never staged in HBM
```

(Real kernels flatten that loop into vectorized index arithmetic rather than
writing it out; the shape of the work is what matters here.)

Read the two versions side by side. FlashAttention computes an address by
*arithmetic*: "tile number 3 starts at 3 × stride." PagedAttention computes it
by *lookup*: "where does tile 3 live? Ask the table." One extra load per tile,
and that is the entire change.

Here is the hop the second version makes, drawn out:

```
   logical position          block table            physical KV blocks
   (what the sequence        (this sequence's       (scattered in HBM,
    thinks it has)            own little map)        any order at all)

   ┌────────────┐           ┌──────┬───────┐        ┌──────────┐
   │ tok  0..15 │  slot 0 ─>│  0   │   93  │───────>│ block 93 │  64 KiB
   ├────────────┤           ├──────┼───────┤        ├──────────┤
   │ tok 16..31 │  slot 1 ─>│  1   │   12  │───────>│ block 12 │  64 KiB
   ├────────────┤           ├──────┼───────┤        ├──────────┤
   │ tok 32..47 │  slot 2 ─>│  2   │   57  │───────>│ block 57 │  64 KiB
   └────────────┘           └──────┴───────┘        └──────────┘
    contiguous, by            4 bytes per            NOT contiguous —
    construction              entry                  and it no longer matters

    per block:  4 B (one index)  ──steers──>  64 KiB of K/V
    per tile :  4 lookups (16 B) ──steers──>  4 blocks = 256 KiB
                                              (BLOCK_N=64, block_size=16)
```

(64 KiB is one 16-token block for *one layer* — `16 × 4,096 B`, derived below.)

The block table itself is tiny, so it stays in fast on-chip memory (**SRAM**,
the small scratchpad next to the compute units) and the lookup costs almost
nothing. Tiny is worth a number: one 4-byte entry per 16 tokens, so even a
32,768-token sequence has a table of

```
32,768 tokens ÷ 16 tokens/block  =  2,048 entries
2,048 entries × 4 B              =  8,192 B  =  8 KiB
```

8 KiB of map to steer megabytes of K/V. The gather disappears entirely.

Note the asymmetry that makes the lookup cheap and the gather expensive. The
gather moved *every byte* of the cached sequence: bytes out of their blocks,
bytes into a duplicate strip, and attention read the duplicate. The block table
lookup moves a few bytes: the address of the next tile's home. That one
address steers a much larger load.

To put a number on "much larger", first be precise about what this kernel is
working on, because it is easy to reach for the wrong figure here. **One
program instance handles one (query tile, head) pair, for one layer.** The
attention kernel runs inside a single layer; it never sees the other 27. So the
K/V cost that matters is the *per-layer* one, not the whole-model one.

Lecture 05 counted both. Per token, in one layer:

```
2 (K and V)  ×  8 KV heads  ×  128 head_dim  ×  2 bytes  =  4,096 B  =  4 KiB
```

and that is the number this kernel spends. (Lecture 05's headline figure,
114,688 B per token, is that same 4 KiB multiplied by all 28 layers — the cost
of one token to the *whole model*. Use it for cache-capacity questions, not for
one kernel's tile.)

So with `block_size = 16`, one **block** holds:

```
16 tokens × 4,096 B/token  =  65,536 B  =  64 KiB
```

and the lookup that finds it is a single 32-bit index, 4 bytes:

```
   4 B  (one block-table entry)   steers   65,536 B  (one block)
   ratio  =  65,536 / 4  =  16,384×
```

One byte of address per 16,384 bytes of payload. Scale it to a whole tile at
`BLOCK_N = 64`, which is four blocks:

```
   16 B  (four entries)   steers   262,144 B  (256 KiB of K/V)
```

Same ratio, four times the payload. That is why the indirection costs almost
nothing while the gather it replaced cost a full extra copy of the sequence:
you spend a handful of address bytes to avoid moving hundreds of kilobytes,
every tile, in every layer, on every step.

Sanity-check the tile against SRAM, because this is where a mistake shows up
loudly. That 4,096 B/token covers K *and* V together, so at `BLOCK_N = 64`:

```
   K tile   64 tokens × 2,048 B  =  128 KiB
   V tile   64 tokens × 2,048 B  =  128 KiB
                                    ────────
                                    256 KiB   against a ~100 KB scratchpad
```

It does not fit — and that is a real constraint, not an arithmetic slip. Paged
decode kernels respond by using a smaller `BLOCK_N`, or by streaming blocks
through rather than holding a whole tile resident. Decode also helps itself
here: its "tile" is a *single* query row rather than 64, so the score patch is
`1 × BLOCK_N` instead of `64 × 64`, and the budget looks nothing like L17's
prefill case.

The general rule: when a tile looks too big for SRAM, either `BLOCK_N` is too
large for this kernel, or you have mixed up per-layer and whole-model K/V — the
114,688 B figure overshoots by 28×.

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

"Nearly idle" deserves a number, because it is the entire reason this section
exists. The natural parallel unit for attention is one thread block per
(sequence, head). At batch 1 with Qwen3-0.6B's 16 query heads, that is
**16 thread blocks** — on a 3090 with **82 SMs**:

```
   work available :  1 sequence × 16 heads  =  16 thread blocks
   machine has    :                            82 SMs

   occupancy      :  16 / 82  ≈  20%      ← 66 SMs with nothing to do
```

Four fifths of the GPU is idle, and no amount of kernel tuning fixes it: there
simply is not enough work to hand out. Prefill never has this problem (thousands
of query tokens, so thousands of tiles), and neither does decode at large batch
— it is specifically the low-batch decode case, which is exactly the latency-
sensitive case you care about.

The fix is to manufacture more parallel work by **splitting the context across
blocks**, each computing a partial (max, sum, accumulator), then combining. Split
a 2048-token context eight ways and the same batch-1 request now offers
`16 × 8 = 128` blocks — more than the machine has SMs, so the GPU fills:

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
   the cached context, split across thread blocks

   ┌──────────────────┬──────────────────┐
   │ tokens    0-1023 │ tokens 1024-2047 │
   └────────┬─────────┴─────────┬────────┘
            │                   │
        block 0             block 1          run independently,
            │                   │            in parallel, on
            ▼                   ▼            different SMs
      (m_0, l_0, acc_0)   (m_1, l_1, acc_1)
            │                   │
            └─────────┬─────────┘
                      ▼
              rescale each by exp(m_i − m_global),
              add, divide by the combined l
                      ▼
                 final output
```

The combination uses the *identical* online-softmax merge from Lecture 17:
rescale each partial by `exp(m_i - m_global)`, sum, divide. Having built that
already, this is a small step rather than a new idea. In words: each block ran
softmax with its own local max `m_i`; the block with the largest max, `m_global`,
had it right, but the others used a max they later learned was too small, so
their accumulators are inflated by `exp(m_i - m_global)`. Shrink each partial
back to what a global max would have produced, add the pieces, divide by the
corrected sum: one attention output, assembled from independent slices.

**Check it on numbers**, the same way Lecture 17's worked example checks the
tile merge. Four cached tokens, `d = 2`, one query, split into two blocks of
two. Scores come out `[1, 3, 2, 0]`:

```
one-shot (what a single block would compute):
   m = 3      l = 1.5530      out = [0.3881, 0.9449]

split across two blocks, each with its own local max:
   block 0   scores [1, 3]   m_0 = 3   l_0 = 1.1353   acc_0 = [0.1353, 1.0000]
   block 1   scores [2, 0]   m_1 = 2   l_1 = 1.1353   acc_1 = [1.2707, 1.2707]

merge:  m_global = max(3, 2) = 3

   block 0 rescale:  exp(3 − 3) = 1.0000    ← it had the right max, untouched
   block 1 rescale:  exp(2 − 3) = 0.3679    ← its max was too small, shrink it

   l   = 1.0000·1.1353 + 0.3679·1.1353            = 1.5530     ✓
   acc = 1.0000·[0.1353,1] + 0.3679·[1.2707,1.2707] = [0.6028, 1.4675]

   out = acc / l = [0.3881, 0.9449]                            ✓ identical
```

Note the shape of it: **the block that happened to hold the largest score is
left alone, and every other block is shrunk.** No block needs to know anything
about the others while it runs — only its own `(m, l, acc)` — and the merge
needs only those three numbers per block, not the scores that produced them.
That is what makes the split embarrassingly parallel.

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

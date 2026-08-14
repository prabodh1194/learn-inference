# 09. Paged attention

**Build:** `engine/block_manager.py::BlockManager`, `engine/cache.py::PagedKVCache`
**Test:** `tests/test_09_paged.py` · **Demo:** `book/code/fragmentation.py`
**Moves:** concurrent sequences before OOM: the biggest single win in Part II
**Prereq:** [08. Continuous batching](08-continuous-batching.md)

> **From here you want a real NVIDIA GPU.** The logic is testable on a laptop,
> and the tests are written that way, but the payoff is a memory number you can
> only see with real VRAM. See [00. Introduction](00-intro.md#hardware).

---

## The problem

Your scheduler wants to admit more requests. Memory says no, long before the GPU
is actually full.

Walk through one admission. Two requests are waiting: a short chat turn that
might produce 128 output tokens, and a long document question that could run to
32k. The scheduler checks: does the KV cache have room for both? The answer is
no, and the reason sits in `KVCache` from Lecture 05:

```python
shape = (max_seqs, max_seq_len, n_kv_heads, head_dim)
```

Look at the second dimension. Every slot is sized for `max_seq_len` tokens
**up front**, whether the sequence that lands there will use one hundred tokens
or thirty thousand. Lecture 05's arithmetic says each cached token costs
112 KiB on this model (2 × 28 layers × 8 KV heads × 128 head_dim × 2 bytes per
token), so a slot that advertises 32k context reserves:

```
32,768 tokens  ×  112 KiB  =  3,670,016 KiB  =  3.5 GiB
```

That is the reservation, not the bill. A 128-token chat request that gets the
slot pays the full 3.5 GiB and then uses

```
128 / 32,768  =  0.39% of the reservation
```

It uses 0.4% of what it holds. And the other 99.6% is not usable slack: it is a
hole held open for a sequence that may never arrive, in a region no other
sequence is allowed to touch.

Why can't the allocation grow on demand? Because the storage is **contiguous**:
the cache is one tensor, and a kernel reads a sequence's rows as a single
unbroken span. A span can only extend into free memory, and the memory next to
it belongs to another sequence's slot. Growing in place means displacing a
neighbor, which means relocating that neighbor's entire cache and copying
everything it holds, on the hot path. Nobody does that, so the design takes the
other route: reserve the worst case, every time.

The result is what `fragmentation.py` shows next: the scheduler's memory
admission test fails on reserved-but-empty space, and the GPU sits partly idle
while requests wait outside.

---

## See it

```bash
uv run python book/code/fragmentation.py
```

On a 24GB 3090 with realistic mixed traffic:

```
  max_seq_len   contiguous     paged     gain
         2048           96       121     1.3x
         4096           48        84     1.8x
         8192           24        84     3.5x
        32768            6        84    14.0x
```

The gain column is one division each: how many more sequences paging fits into
the same KV budget:

```
121 / 96  =  1.26x      84 / 48  =  1.75x
 84 / 24  =  3.5x       84 / 6   =  14.0x
```

**Six sequences at 32k context. Six.** And where that memory goes (this block
is `max_seq_len=8192`):

```
  sequences fitted     24
  tokens reserved      196,608        = 24 sequences x 8192 max tokens
  tokens actually used  58,249
  WASTED               70.4%
```

The waste fraction: `(196,608 − 58,249) / 196,608 = 138,359 / 196,608 = 70.4%`.
More than two-thirds of the reservation is empty space.

Note the shape of the table: the longer the context you *support*, the worse
contiguous allocation gets, while paged stays flat at 84. Contiguous is punished
for capability you're not even using.

---

## The idea

The fix is the oldest trick in operating systems: **virtual memory**.

An OS doesn't give a process one contiguous block of physical RAM. It hands out
fixed-size **pages** and keeps a **page table** mapping virtual addresses to
physical ones. Processes see a clean contiguous space; physically it's scattered.

PagedAttention (Kwon et al., 2023) applies this to the KV cache:

- Carve VRAM into fixed-size **blocks**, say 16 tokens each.
- Give each sequence a **block table**: logical position → physical block.
- Allocate a new block only when the sequence actually needs one.

```
Sequence A (37 tokens)   block_table = [4, 1, 9]
                                        ↓  ↓  ↓
physical blocks:  [0][1][2][3][4][5][6][7][8][9]...
                      A2       A0          A1

  block 4 -> tokens 0-15
  block 1 -> tokens 16-31
  block 9 -> tokens 32-37  (partially used -- 10 tokens of slack)
```

Blocks need not be adjacent. Attention gathers them through the table.

### What this changes

**Waste becomes bounded.** Contiguous allocation wastes `max_seq_len - actual`
per sequence, unbounded, and growing with the context you advertise. Paging
wastes at most `block_size - 1` tokens: *internal fragmentation only*.

**Growth is incremental.** A sequence gets one more block when it crosses a
boundary, not a reservation at admission time.

**Sharing becomes possible.** Two sequences can point at the *same* physical
block. That's Lecture 10, and it falls out of this design for free.

### Choosing block size

From the demo:

```
  block_size   sequences   waste/seq   blocks/seq
           1          84        0.0t         2062
           8          84        3.5t          258
          16          84        7.7t          129
          32          83       15.6t           65
         128          83       63.0t           17
         512          77      253.0t            5
```

Read the columns, not the row order. Block sizes 1 and 16 fit the *same* 84
sequences: the memory saved by finer blocks is already negligible by 16. What
differs is the block table: **2062 entries per sequence at block 1, versus 129 at
16**, every one an indirection on the hottest path in the system.

Only at 512 does waste start costing you real capacity (77 sequences, 253 tokens
wasted each). **vLLM defaults to 16** because that's where waste has gone to
nothing and the table is still short.

### The part that isn't free

Attention can no longer read a contiguous span. Every step it must gather K/V
through the block table, an indirection on the hottest path in the system.

This is why PagedAttention needs a **custom kernel**. You can prototype it with
gathers in PyTorch (do that now), but the production version is hand-written
CUDA. That's Lecture 18, and you'll be glad you already know what the block table
means.

---

## The code

```python
class BlockManager:
    def __init__(self, n_blocks, block_size=16, enable_prefix_caching=False):
        self.block_size = block_size
        self.free_blocks: deque[int] = deque(range(n_blocks))
        self.ref_counts: dict[int, int] = {}      # Lecture 10 needs this

    def blocks_needed(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)     # ceil

    def can_allocate(self, seq) -> bool:
        return self.blocks_needed(len(seq)) <= len(self.free_blocks)

    def allocate(self, seq) -> list[int]:
        need = self.blocks_needed(len(seq)) - len(seq.block_table)
        for _ in range(need):
            block = self.free_blocks.popleft()
            self.ref_counts[block] = 1
            seq.block_table.append(block)
        return seq.block_table

    def append_token(self, seq) -> None:
        """Called each decode step. Usually a no-op -- only 1 step in
        block_size actually crosses a boundary."""
        if len(seq) > len(seq.block_table) * self.block_size:
            block = self.free_blocks.popleft()
            self.ref_counts[block] = 1
            seq.block_table.append(block)

    def free(self, seq) -> None:
        for block in seq.block_table:
            self.ref_counts[block] -= 1
            if self.ref_counts[block] == 0:        # nobody else points here
                self.free_blocks.append(block)
        seq.block_table.clear()
```

**Reference counting from the start.** You don't need it yet, every block has
exactly one owner. Lecture 10 makes blocks shared, and retrofitting refcounts
into a design that assumed sole ownership is unpleasant. Build the hook now.

The gather, in PyTorch, for prototyping:

```python
def gather_kv(cache, block_table, seq_len, block_size):
    """Reassemble a logical K/V sequence from scattered physical blocks."""
    blocks = torch.tensor(block_table, device=cache.device)
    gathered = cache[blocks]                       # (n_blocks, block_size, H, D)
    flat = gathered.reshape(-1, *cache.shape[2:])  # (n_blocks*block_size, H, D)
    return flat[:seq_len]                          # trim the partial last block
```

Correct and slow, it materializes the whole sequence. Fine for now; Lecture 18
fuses it into the attention kernel so nothing is materialized at all.

### Preemption

You can now answer the question Lecture 08 deferred. Out of blocks with requests
waiting? Evict a running sequence:

- **Swap**: copy its blocks to host RAM, restore later. Costs PCIe bandwidth.
- **Recompute**: drop the blocks, redo prefill on resume. Costs compute.

vLLM does both, choosing by sequence length. Short sequences are cheap to
recompute; long ones are cheaper to swap.

---

## Build it

1. Implement `BlockManager` in `engine/block_manager.py`.
2. Implement `PagedKVCache` in `engine/cache.py`.
3. Wire `can_allocate` into the scheduler's admission check from Lecture 08.
4. `uv run pytest tests/test_09_paged.py -v`, **paged output must match
   contiguous exactly.** This is a storage change; the model must not notice.
5. Measure the number that matters:

```bash
uv run python book/code/paged_bench.py --max-concurrent
```

Push concurrent sequences up until OOM, with and without paging. **Record both.**

---

## What you should see

**Substantially more concurrent sequences**: how many depends on your
`max_seq_len` and traffic, but the gap widens as supported context grows.

**Throughput up as a consequence.** More concurrent sequences means bigger decode
batches, and from Lecture 01 bigger batches directly raise arithmetic intensity
on a memory-bound phase. A memory optimization bought you throughput.

**Per-step latency slightly worse.** The gather isn't free, especially in PyTorch.
Lecture 18 wins most of it back.

---

## Go deeper

- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)**
  (Kwon et al., SOSP '23): the vLLM paper. **Read it now**, having just built
  the thing. §4 is the memory manager; the OS analogy is drawn explicitly.
- **vLLM `vllm/v1/core/block_pool.py`** and **`kv_cache_manager.py`**, the
  production version of what you wrote.
- **nano-vllm `nanovllm/engine/block_manager.py`**: ~4.3KB, much closer to yours.
- **Kiely §2.5** (p.68), PagedAttention in context.
- **Kiely §5.3.2** (p.139): the G1–G4 storage hierarchy; where swapped blocks go.

---

## Check yourself

1. Contiguous fits 6 sequences at 32k context and 96 at 2k. Paged fits ~84 at
   both. Why is paged flat?
2. Why is paged waste bounded by `block_size - 1` per sequence, while contiguous
   waste is unbounded?
3. Block size 1 would waste nothing at all. Why doesn't anyone use it?
4. Paging is a memory optimization. Explain, via Lecture 01, why it raises
   *throughput*.
5. Two sequences share a 400-token system prompt. What would let them share
   physical blocks, and what must be true before you dare?

That last one is the next lecture.

---

## Next

**[10. Prefix caching](10-prefix-caching.md)**: blocks can be shared, and the
savings are larger than you'd guess.

```bash
uv run python book/code/prefix_bench.py
```

It runs `shared_prefix` and `late_divergence` back to back, near-identical
token counts, opposite cache behaviour. **Predict the TTFT for each before you
run it.**

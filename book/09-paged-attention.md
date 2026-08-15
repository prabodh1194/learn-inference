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

Start with the person at the door. Every engine has a **scheduler**: it decides
which requests get in, and in what order. Right now two requests are waiting: a
chat that will probably reply in 128 tokens, and a long document question that
could run to 32,000. The scheduler asks the same question before anything
else: is there room?

To answer that you need to know what "room" means here. The model does not
remember anything on its own. As the conversation proceeds, it writes a running
record into the **KV cache** from Lecture 05, a scratchpad that stores what
each token offered the tokens after it. The scratchpad has a fixed price per
token, 112 KiB on our model (2 copies, K and V, × 28 layers × 8 heads × 128
values × 2 bytes):

```python
shape = (max_seqs, max_seq_len, n_kv_heads, head_dim)
#        how many   how long     how wide each
#        requests   a single     token's entry is
#        fit        request may
#                   grow to
```

Prices are easy to underestimate: a 2,000-token chat costs 224 MiB, a 32k
conversation costs 3.5 GiB, and that is on a model whose weights are only
840 MiB. The cache is the bigger of the two costs on any long conversation.

Now the scheduler comes to the request. How much will this chat cost? And here
is the catch: **it cannot know.** The chat might stop in 20 tokens or keep
talking for 30,000. The engine's rule is therefore "admit at the maximum":
every request is booked as if it will use all 32,768 tokens, no matter what it
actually says back.

```
reserved, for every admitted request:   32,768 × 112 KiB  =  3.5 GiB
used,      by the 128-token chat:          128 × 112 KiB  =  14 MiB
the chat actually uses:        14,336 / 3,670,016  =  0.39%  of its booking
```

Draw it. The cache is a grid: each row belongs to one admitted request, and
each column is one reserved token slot. When a request arrives it is handed a
whole row, even though it will only ever sit in a few columns:

```
row A = the 128-token chat (a 32,768-token row, drawn shortened)

  [====][..............................................................]
   128    32,640 columns sitting empty
   used
          they are empty, but no other request may touch them.
          they belong to row A for as long as row A exists.

row B = the long document request (same story)
  [==================][.......................empty, reserved..............]
   already used, will grow         held in case it does
```

Then a third request arrives and the scheduler says: no room. The machine is
nowhere near full: row A alone is 99.6% empty. But empty is not free. Row A's
booking stands until the chat actually ends, and that booking is what counts
for the memory test.

Why can't the chat's row just grow as the chat gets longer, borrowing columns
from row B? Because a row is one **continuous strip** of memory, and the chip
reads it as a single block, all in one go. A strip can only grow at its end,
into the space immediately after it, and that space is already assigned to
row B. The alternative is to copy row A to a bigger empty area somewhere else
every time it grows, and copying gigabytes is far too slow to happen between
tokens of a live conversation (each arrives in milliseconds). So the engine
reserves the worst case up front, every time.

There is a third way memory disappears, and it needs a slightly different
admission policy. Suppose the engine is honest instead: the chat is booked at
its real cap of 128 tokens, the document at its 32,768. When the chat
finishes, its strip joins the free list — as a **128-token strip**. The next
long request still needs 32,768 contiguous tokens, and a 128-token strip can
never serve it. Memory that is genuinely free but unusable, because the free
pieces are the wrong sizes for every waiting request: **external
fragmentation**. Uniform bookings (everyone at the full maximum) dodge this
exactly because they are uniform — and pay for it with the reservation waste
above. Contiguous allocation cannot have both: the pieces are always the wrong
shape for something.

??? question "How can memory be free but unusable?"
    Freed strips come in the size of the request that freed them. A finished
    128-token chat leaves a 128-token strip, and a 32,768-token document
    cannot live in it. Free space is only usable when a waiting request needs
    a contiguous strip that size — otherwise it is dead until some neighbour
    frees enough. That is external fragmentation, and paging eliminates it:
    every block is the same size, so a freed block fits any next request.
    [Full answer](qa.md#how-can-memory-be-free-but-unusable)

> That mismatch is the whole problem of this lecture: small requests, huge
> reservations, memory filling with empty space, and a machine that could
> serve everyone saying no to everyone. The next section counts exactly how
> much gets turned away.

??? question "Why can't the cache just grow as the conversation grows?"
    Because a sequence's cache must stay one continuous strip of memory: the
    chip reads it as a single block. A strip can only extend into space right
    after it, which belongs to the next sequence. Growing = copying everything
    to a bigger location, too slow for live tokens. The fix is to give up on
    contiguity entirely. [Full answer](qa.md#why-cant-the-sequences-cache-just-grow-as-it-needs-space)

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
wastes at most `block_size - 1` tokens: *internal fragmentation only*. External
fragmentation disappears with it — every block is the same size, so a freed
block serves any next request.

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

The kernel adds its own floor from below (Lecture 18): the 16 tokens inside a
block sit in contiguous memory, so one block reads as one wide, coalesced
burst. Too small, and decode attention pays a scatter jump per block and
reads under-use the bus; too big, and the slack of the final block grows back.
Block size is bounded on both sides; 16 is the middle where both the memory
and the kernel are happy.

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

    # `cache` is the whole pool of physical blocks, one big tensor:
    #   shape (n_blocks_total, block_size, H, D)
    #     n_blocks_total  -- every block in the pool, allocated or free
    #     block_size      -- tokens per block (16 by default)
    #     H               -- KV heads
    #     D               -- head dimension
    #
    # `block_table` is one sequence's *logical* order: a list of physical
    # indices, e.g. [3, 7, 12] means "my tokens live in block 3, then 7,
    # then 12" -- in that order, no matter where those blocks sit in cache.

    # A Python list can't index a GPU tensor. Turn it into a tensor first,
    # and put it on the same device as `cache` (indices and data must be
    # on the same device, CPU with CPU, GPU with GPU).
    blocks = torch.tensor(block_table, device=cache.device)

    # Fancy indexing: `cache[blocks]` picks out the *rows* of cache whose
    # indices are in `blocks`, stacked in that same order. Every other
    # dimension comes along untouched, so the result has shape
    #   (len(block_table), block_size, H, D)
    # This one line is the whole "reassembly": the scattered blocks become
    # one contiguous tensor, in logical order.
    gathered = cache[blocks]

    # Collapse the first two dims -- (n_blocks, block_size) -- into a single
    # token dimension. Before: (n_blocks, block_size, H, D). After:
    #   (n_blocks * block_size, H, D)
    # `-1` means "figure out this size" (= n_blocks * block_size). The
    # `*cache.shape[2:]` unpacking just says "keep (H, D) as-is".
    flat = gathered.reshape(-1, *cache.shape[2:])

    # The last block is usually only partly full: the sequence stopped
    # mid-block. Keep only the `seq_len` tokens that actually exist.
    return flat[:seq_len]
```

Concretely, with `block_table = [3, 7, 12]`, `block_size = 16`, and a 40-token
sequence (`seq_len = 40`) on a pool of 1000 blocks:

```
cache           (1000, 16, 8, 128)   the whole pool
cache[blocks]   (3,    16, 8, 128)   blocks 3, 7, 12, in logical order
flat            (48,   8,  128)      3 × 16 = 48 tokens
flat[:seq_len]  (40,   8,  128)      drop the 8 padding tokens in block 12
```

Correct and slow, it materializes the whole sequence. Fine for now; Lecture 18
fuses it into the attention kernel so nothing is materialized at all.

### Preemption

You can now answer the question Lecture 08 deferred. Out of blocks with requests
waiting? Evict a running sequence. Two constraints shape how:

**Eviction is all-or-nothing per sequence.** A sequence cannot take a step with
half its blocks — the single new token attends over everything it has cached —
so evicting a few blocks helps nobody. You evict a whole sequence or none.

**The victim is the newest arrival.** vLLM admits in arrival order (FCFS) and
preempts in reverse: the most recently admitted sequence is swapped out first,
letting the oldest ones — nearest to finishing — run to completion and free
their blocks. Then the victims resume, oldest first.

Two ways to resume a victim:

- **Swap**: copy its blocks to host RAM, restore later. Costs PCIe bandwidth.
- **Recompute**: drop the blocks, redo prefill on resume. Costs compute.

Which wins is an ablation, not a taste. With small blocks, swapping ships one
tiny chunk per transfer and the PCIe overhead dominates: recompute wins. With
large blocks the transfer is a fat contiguous read and swap wins. Sequence
length enters too: a long sequence is expensive to recompute, so long victims
get swapped, short ones recomputed. vLLM implements both and picks per
situation — and in its measurements neither path is the bottleneck.

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

The paper reports the same story on a bigger stage: 2–4× the throughput of
FasterTransformer and Orca on identical hardware, sustained at equal latency,
across OPT-13B/66B/175B on real traces (ShareGPT, long and chatty; Alpaca,
short and uniform) — and the gap is largest on exactly the traffic where KV
memory is tightest. One baseline is Orca with perfect foresight, told each
request's true output length at admission, and paging still wins: no
reservation policy, however well-informed, beats not reserving.

**Per-step latency slightly worse.** The gather isn't free, especially in PyTorch.
Lecture 18 wins most of it back.

---

## Go deeper

- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)**
  (Kwon et al., SOSP '23): the vLLM paper. **Read it now**, having just built
  the thing. §4 is the memory manager; the OS analogy is drawn explicitly.
- **["What is vLLM? | PagedAttention | Fully Explained: an OS Trick for 4× Throughput"](https://www.youtube.com/watch?v=xgl9Qrz31Mc)**
  (Papers by Hand): a by-hand walkthrough of the same paper — the three kinds
  of waste, a decode traced block-by-block, copy-on-write, preemption, and
  where the 2–4× number comes from.
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

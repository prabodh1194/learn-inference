# 08. Continuous batching

**Build:** `engine/scheduler.py::Scheduler`, `engine/sequence.py` · **Test:** `tests/test_08_scheduler.py`
**Moves:** throughput up ~2.5× on mixed load; slot waste 61% → ~0%
**Prereq:** [07. Static batching](07-static-batching.md)

---

## The problem

You measured it: static batching wastes **61% of decode slots** on realistic
traffic, and the waste gets *worse* as batches get bigger.

The cause is structural, not a tuning issue. A static batch is fixed for its
lifetime, a sequence that finishes at step 8 holds its slot until step 512
because the batch cannot change shape mid-flight.

So make it change shape mid-flight.

---

## The idea

> **Schedule per step, not per batch.** When a sequence finishes, evict it and
> admit a waiting one, immediately, at the next step boundary.

The batch stops being a fixed group and becomes a *sliding window* over a queue.
Idle slots are refilled instead of held. This is **continuous batching**
(sometimes "in-flight batching"), introduced in the Orca paper.

The insight it rests on is one you already proved in Lecture 01: **decode steps
are memory-bound**, so adding a sequence to an in-flight batch is nearly free.
The weights were being loaded anyway. An empty slot isn't saving you anything,
it's pure waste.

### The architectural consequence

You cannot express this as a loop around `generate()`. The function has to be
turned inside out:

```
BEFORE                          AFTER
generate(prompts):              scheduler.schedule()  -> which requests run now
  loop:                         runner.step(batch)    -> execute exactly one step
    forward everything          scheduler.update()    -> retire, admit, repeat
```

Two components:

- **Scheduler**: owns the queue. Decides which requests are in this step's batch,
  handles admission, completion, and preemption.
- **Model runner**: stateless. Given a batch, executes one forward pass.

**This split is the architecture of every serving engine in existence.** vLLM has
it (`vllm/v1/core/sched/scheduler.py`), so does nano-vllm, so does TensorRT-LLM.
After this lecture, their source stops looking foreign.

### Requests need identity now

Once sequences enter and leave independently, "the batch" is no longer a
meaningful unit of state. Each request carries its own:

```python
@dataclass
class Sequence:
    seq_id: int
    prompt_ids: list[int]
    output_ids: list[int]
    status: Status              # WAITING / RUNNING / PREEMPTED / FINISHED
    max_tokens: int
    # Lecture 09 adds: block_table -- where this sequence's KV cache lives
```

### What the scheduler decides

Each step, three questions:

1. **Who's finished?** Hit EOS or `max_tokens` → free their resources.
2. **Who's still running?** They need one decode step.
3. **Can anyone new be admitted?** Room in the batch *and* memory for their KV
   cache → admit and prefill them.

Two budgets bound the answer:

- **`max_batch_size`**: concurrent sequences.
- **`max_batched_tokens`**: total tokens per step. A prefill of 4,000 tokens is
  much more work than 32 decode steps, so counting sequences alone underestimates
  a step's cost badly.

### The prefill/decode conflict

Here's the tension that Lecture 11 exists to resolve. Admitting a new request
means running its prefill, which is expensive and *compute-bound*. Meanwhile
every running sequence wants a cheap *memory-bound* decode step.

Mix them and prefill dominates the step, every decoding user stalls. This is why
naive continuous batching improves throughput while making **p99 latency worse**.

Watch for that in your measurements. It's a real effect, and noticing it yourself
is what makes chunked prefill feel necessary rather than arbitrary.

---

## The code

```python
class Scheduler:
    def __init__(self, max_batch_size=32, max_batched_tokens=8192, ...):
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        """Return (to_prefill, to_decode) for exactly one step."""
        # 1. retire anything finished -- frees slots AND memory
        for seq in list(self.running):
            if seq.is_finished():
                self.running.remove(seq)
                self.free(seq)

        # 2. everything still running gets one decode step
        decode = list(self.running)
        budget = self.max_batched_tokens - len(decode)   # 1 token each

        # 3. fill remaining capacity with new admissions
        prefill = []
        while self.waiting and len(self.running) < self.max_batch_size:
            seq = self.waiting[0]
            need = len(seq.prompt_ids)
            if need > budget:
                break                     # doesn't fit this step; try next
            if not self.can_allocate(seq):
                break                     # out of KV memory -- Lecture 09
            self.waiting.popleft()
            self.allocate(seq)
            seq.status = Status.RUNNING
            self.running.append(seq)
            prefill.append(seq)
            budget -= need

        return prefill, decode
```

Details worth noting:

**Retire first.** Freeing finished sequences before admitting new ones is what
makes the slot available in the *same* step. Do it in the other order and you
add a step of latency to every admission.

**`break`, not `continue`.** When the head of the queue doesn't fit, stop. Skipping
past it to admit a smaller request behind is head-of-line jumping, it starves
long requests indefinitely under load. Simple FIFO is fair and predictable;
deviate deliberately, not accidentally.

**Preemption.** Under memory pressure you may need to evict a *running* sequence.
Two options: swap its KV cache to host memory, or discard and recompute later.
vLLM does both. You can defer this until Lecture 09 gives you block-level
accounting, but note the hook now.

---

## Build it

1. Implement `Sequence` and `Status` in `engine/sequence.py`.
2. Implement `Scheduler.schedule()` in `engine/scheduler.py`.
3. Restructure generation into `scheduler.schedule()` → `runner.step()` → repeat.
   This is a real refactor, not a patch, expect to move code around.
4. `uv run pytest tests/test_08_scheduler.py -v`, scheduler logic is tested
   **without the model**, so iterate fast on the hard part.
5. Measure on the workload that shows it:

```bash
uv run python book/code/batch_bench.py
```

**Compare against Lecture 07 on `mixed_length`.** Record throughput, slot waste,
**and p99 latency**. Predict the p99 direction before you look.

---

## What you should see

**Throughput up ~2–2.5×** on `mixed_length` (the demo predicted 2.57× for an ideal
scheduler; you'll be under that, and the gap is your scheduling overhead).

**Slot waste near zero.** Sequences no longer wait on each other.

**On `uniform`, almost no improvement**: nothing was being wasted, so nothing was
recovered. If that surprises you, re-read Lecture 07's demo.

**p99 latency possibly worse.** Prefills interleaving with decodes cause stalls.
That's Lecture 11.

---

## Go deeper

- **[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)**
  (Yu et al., OSDI '22): the source. "Iteration-level scheduling" is their term
  for what you just built. §3 is the core.
- **vLLM `vllm/v1/core/sched/scheduler.py`**: read `schedule()` now. It's your
  function plus chunked prefill, prefix caching, preemption, and specdec. You'll
  recognize the skeleton.
- **nano-vllm `nanovllm/engine/scheduler.py`**: ~3.7KB, much closer to what you
  just wrote. Good for a direct diff.
- **Kiely §7.2.1** (p.186), concurrency and batch sizing in production.
- **[Field notes](field-notes.md)**: 100 tok/s single-user vs. 585 tok/s across 8.

---

## Check yourself

1. Why is adding a sequence to an in-flight decode batch nearly free? *(One
   sentence, from Lecture 01.)*
2. Continuous batching barely helps on `uniform` and helps enormously on
   `mixed_length`. Why?
3. Your throughput rose and p99 got worse. What's causing that, and what would
   you change?
4. Why `break` rather than `continue` when the queue head doesn't fit? What
   breaks if you skip ahead?

---

## Next

**[09. Paged attention](09-paged-attention.md)**: your scheduler now wants to
admit more requests, and memory says no.

```bash
uv run python book/code/fragmentation.py    # run before reading
```

> **This is where a GPU starts to matter.** The block logic is testable on a
> laptop and the tests are written that way, but the payoff is a capacity
> number you can only see with real VRAM. See [00. Introduction](00-intro.md)
> for rental notes.

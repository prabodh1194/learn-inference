# 07. Static batching

**Build:** `engine/generate.py::generate_batched` · **Test:** `tests/test_07_batching.py`
**Demo:** `book/code/batching_waste.py` · **Moves:** aggregate throughput up several ×; per-user latency **worse**
**Prereq:** [06. Sampling](06-sampling.md)

---

## The problem

You can serve one request efficiently. Now serve thirty-two.

The naive approach (one at a time) leaves the GPU almost entirely idle. Lecture
01 showed why: decode reloads all 840 MiB of weights to generate one token. Those
same weights could have served 32 sequences for the same memory traffic.

So batch them. It works, and it introduces a new problem that takes the next
lecture to solve.

---

## The idea

Stack sequences into a batch dimension and run them through the model together:

```
one at a time:  (1, seq, hidden)  ×32 forward passes
batched:        (32, seq, hidden) ×1  forward pass
```

The weights load **once** for all 32. That's the whole win, and from Lecture 02's
table it's close to linear in batch size while you're memory-bound.

But sequences have different lengths, and a tensor is rectangular. So you pad:

```
"Explain paged attention"     -> [1234, 5678, 910, PAD, PAD, PAD, PAD, PAD]
"Hi"                          -> [4321, PAD, PAD, PAD, PAD, PAD, PAD, PAD]
"Write a function that ..."   -> [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
```

Padding is computed and discarded, an attention mask keeps it from affecting
results, but the GPU does the work regardless.

### The part that actually hurts

Padding waste is the obvious cost. The expensive one is subtler:

> A static batch runs until its **longest** member finishes. Sequences that
> complete early hold their slots, doing nothing, until the whole batch retires.

Ask for 8 tokens while someone else asks for 512, and your slot is idle for 504
steps. It cannot be reused, because the batch is fixed for its lifetime: that's
what "static" means.

---

## See it

```bash
uv run python book/code/batching_waste.py
```

Two workloads, same code:

```
uniform (all requests identical)
  prefill padding waste       0.0%
  decode slot waste           0.0%
  -> continuous batching is 1.00x better here

mixed_length (realistic)
  prefill padding waste      64.0%
  decode slot waste          61.0%
  -> continuous batching is 2.57x better here
```

**Zero waste on uniform load. 61% on realistic load.**

And the trap in the batch-size sweep:

```
  batch   decode waste   vs continuous
      1           0.0%           1.00x
      8          61.0%           2.57x
     16          65.9%           2.93x
```

Bigger batches do more useful work per weight load **and** waste more slots,
simultaneously. Static batching cannot escape this; the two move together.

> **Note what the uniform row implies.** If you benchmark only on uniform load,
> static batching looks as good as anything else and you'd conclude the next
> lecture is unnecessary. This is the most common way to fool yourself when
> evaluating an inference optimization, and it's why `bench/workloads.py`
> documents which workload exposes which win.

---

## The code

```python
def generate_batched(model, tokenizer, prompts, max_tokens=128, on_token=None):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    ids = enc.input_ids.to(model.device)
    mask = enc.attention_mask.to(model.device)

    with torch.no_grad():
        out = model(ids, attention_mask=mask, use_cache=True)
    past = out.past_key_values
    next_ids = out.logits[:, -1].argmax(-1, keepdim=True)

    generated = [next_ids]
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=ids.device)

    for _ in range(max_tokens - 1):
        mask = torch.cat([mask, (~finished).long().unsqueeze(-1)], dim=-1)
        with torch.no_grad():
            out = model(next_ids, attention_mask=mask, past_key_values=past,
                        use_cache=True)
        past = out.past_key_values
        next_ids = out.logits[:, -1].argmax(-1, keepdim=True)

        finished |= next_ids.squeeze(-1) == tokenizer.eos_token_id
        generated.append(next_ids)
        if on_token:
            on_token()                      # one call per STEP, not per sequence
        if finished.all():
            break                           # everyone done -- rare on mixed load

    return decode_each(tokenizer, torch.cat(generated, dim=-1), finished)
```

**`padding_side="left"`** is not optional for decode. With right-padding, the last
position of a short sequence is a pad token, so `logits[:, -1]` predicts from
padding, garbage. Left-padding puts every sequence's real final token at
position −1. This bug produces plausible-looking wrong output, which is the worst
kind.

**The `finished` mask** stops a completed sequence from contributing new tokens.
Its slot is still computed: that's the waste we're measuring.

---

## Build it

1. Implement `generate_batched` in `engine/generate.py`.
2. `uv run pytest tests/test_07_batching.py -v`, **batched greedy output must
   match single-sequence greedy exactly.** Batching is an optimization; it must
   not change results. Padding leaking through the mask is the usual culprit
   when this fails.
3. Measure both workloads:

```bash
uv run python book/code/batch_bench.py
```

4. **Record the padding-waste fraction**, not just throughput. That number is
   what Lecture 08 eliminates, and you want the before.

**Predict first:** at batch 8 on `mixed_length`, what happens to *per-user*
latency versus batch 1? Write it down.

---

## What you should see

**Aggregate throughput up substantially.** On `uniform`, close to linear in batch
size while you have memory bandwidth to spare.

**Per-user latency worse.** Your request now waits for a batch to form and shares
a slower step. This is the fundamental trade, Lecture 25 plots it as a curve.

**A gap between workloads.** `uniform` will look great, `mixed_length` much less
so. That gap is the next lecture.

---

## Go deeper

- **Kiely §7.2.1** (p.186), concurrency and batch sizing as a production knob.
- **[Field notes](field-notes.md)**: a 2×3090 setup: **~100 tok/s for one user,
  585 tok/s across 8**. Aggregate up ~6×, per-user roughly flat. Exactly this
  trade, measured in the wild.
- **[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)**
  (Yu et al., OSDI '22), introduced continuous batching. **Read §2–3 now**: it
  motivates the problem you just measured. Save the rest for Lecture 08.

---

## Check yourself

1. Static batching wastes 0% on `uniform` and 61% on `mixed_length`. Same
   algorithm. What does that tell you about any benchmark you're shown?
2. Batch 8, seven requests want 16 tokens and one wants 512. How many slot-steps
   are wasted? What fraction?
3. Why does `padding_side="left"` matter for decode but not for prefill-only
   scoring?
4. Larger batches raise both useful work per weight load and slot waste. What
   would let you keep the first without the second?

That last one is the next lecture.

---

## Next

**[08. Continuous batching](08-continuous-batching.md)**: the first real
architectural change, and the one that makes this an engine.

**Budget more time for this one.** `generate()` turns inside out into a
scheduler plus a model runner; it's a genuine refactor, not a patch.

The good news: its 10 tests need no model, so you can iterate in milliseconds.

```bash
uv run pytest tests/test_08_scheduler.py -v
```

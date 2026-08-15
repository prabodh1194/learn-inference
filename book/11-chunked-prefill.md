# 11. Chunked prefill

**Build:** chunked admission in `engine/scheduler.py` · **Test:** `tests/test_11_chunked.py`
**Moves:** **p99 latency**, and barely touches the mean
**Prereq:** [10. Prefix caching](10-prefix-caching.md)

---

## The problem

Set the scene with a concrete moment. A service is decoding **22 short chats**,
one token per sequence per step — the steady, boring work Lecture 08 made cheap.
Then someone submits a **4,000-token** document, and the scheduler admits it.

The step that runs that document's prefill is enormous. Count the arithmetic:
every token, prefill or decode, costs one full pass over the weights,
`2 × 440.4M` flops. The prefill adds 4,000 passes to the step, against the 22
decode passes the chats still make (the document is prefilling in a 23rd slot,
so it is not in the decode part):

```
prefill's extra work:   4,000 × 880.8 MFLOP  =  3,523.2 GFLOP
decode part:               22 × 880.8 MFLOP  =     19.4 GFLOP
ratio:  3,523.2 / 19.4  =  182×
```

One **182× step** sitting among normal ones, and there is no spare capacity to
absorb it: prefill is *compute-bound*, arithmetic is the limit (Lecture 02),
unlike decode's memory-bound arithmetic, which hides inside the time the chip
spends waiting on memory anyway. Every sequence decoding alongside it waits for
the whole thing to finish.

```
step 41: [decode ×22] + [prefill 4000 tokens]   <- everyone stalls
step 42: [decode ×23]                            <- normal
```

From the user's side, their token stream just froze for a beat because somebody
else submitted a long document. One slow item at the front of a checkout line
holds up everyone behind it; in scheduling that has a name, **head-of-line
blocking**, and this is it, inside a single step. Do that at p99 scale and
your service feels erratic even though its average is fine: most steps are
healthy, the damage hides in the tail, which is exactly why the mean is a poor
way to evaluate a scheduler.

---

## The idea

Don't run the prefill all at once. **Split it into chunks and spread it across
steps**, interleaved with decode:

```
BEFORE                              AFTER  (chunk = 512)
step 41: 22 decode + 4000 prefill   step 41: 22 decode + 512 prefill
step 42: 23 decode                  step 42: 22 decode + 512 prefill
                                    ...
                                    step 48: 22 decode + 512 prefill
                                    step 49: 23 decode
```

Same total prefill work, spread over `4,000 / 512 = 7.8` → 8 steps instead of
concentrated in one. No
step is ever catastrophically long, so no decoding user sees a long stall.

### Why this is nearly free

The reason it works comes straight from Lecture 01.

Decode is **memory-bound**: the GPU has spare arithmetic capacity while it waits
on weights. Prefill is **compute-bound**. Put a slice of prefill into a decode
step and it largely fills capacity that was going unused.

You are not stealing from decode; you're using the idle arithmetic units that a
memory-bound step leaves lying around. That's why chunked prefill can improve p99
substantially while barely denting throughput.

### The token budget becomes the real knob

`max_batched_tokens` from Lecture 08 now does the actual work:

```python
budget = max_batched_tokens - n_decode_tokens   # decode gets priority
chunk = min(budget, remaining_prompt_tokens)
```

Decode tokens are counted first (running users are protected) and prefill fills
whatever's left. Chunk size is the tuning dial:

| Chunk size | p99 | Throughput | Why |
|---|---|---|---|
| tiny (64) | best | worse | per-step overhead dominates; prefill loses its parallelism |
| **medium (512)** | good | good | the usual sweet spot |
| huge (8192) | bad | best | you've re-created the original problem |

Chunk too small and prefill stops being efficient, its whole advantage was doing
many tokens at once. Chunk too large and you're back to stalling.

### State this needs

A partially-prefilled sequence must remember where it stopped. That's what
`Sequence.num_prefilled` is for, already in `engine/sequence.py`:

```python
@property
def is_prefill_done(self) -> bool:
    return self.num_prefilled >= len(self.prompt_ids)
```

It also composes with Lecture 10: a prefix cache hit *initializes* `num_prefilled`
to the hit length, so cached tokens are skipped and only the remainder gets
chunked.

---

## The code

`schedule()` returns `(prefill, decode)`: which sequences take a *prefill chunk*
this step, and which take their one-token *decode*. Three conventions make the
arithmetic work, and each is easy to miss:

- **A decode is exactly one token.** Every running sequence produces one token
  per step, so the *number of decoding sequences* equals the *number of decode
  tokens*. That's why `budget = max_batched_tokens - len(decode)`.
- **A prefill is a chunk.** A prefilling sequence consumes up to `chunk_size`
  tokens this step (fewer if that's all that's left of its prompt, or all the
  budget allows).
- **One step, one forward pass over everything.** Decodes and prefill chunks
  share one pass, capped by `max_batched_tokens`. The runner reads each
  sequence's `chunk_size_this_step` to know how many prompt tokens to feed it.

```python
def schedule(self):
    # 1. retire finished sequences (Lecture 08)
    ...

    # 2. running sequences decode -- they get the budget first.
    #    Each decode is exactly 1 token, so sequence count == token count.
    decode = [s for s in self.running if s.is_prefill_done]
    budget = self.max_batched_tokens - len(decode)

    # 3. continue in-flight partial prefills BEFORE admitting new work.
    prefill = []
    for seq in self.running:
        if seq.is_prefill_done or budget <= 0:
            continue
        remaining = len(seq.prompt_ids) - seq.num_prefilled
        chunk = min(remaining, budget, self.chunk_size)   # 3-way min
        seq.chunk_size_this_step = chunk
        seq.num_prefilled += chunk
        budget -= chunk
        prefill.append(seq)

    # 4. only then admit new sequences into what's left -- they also start
    #    prefilling, chunked exactly like everyone else.
    while (self.waiting and len(self.running) < self.max_batch_size
           and budget > 0):
        seq = self.waiting.popleft()
        self.running.append(seq)
        remaining = len(seq.prompt_ids)
        chunk = min(remaining, budget, self.chunk_size)
        seq.chunk_size_this_step = chunk
        seq.num_prefilled += chunk
        budget -= chunk
        prefill.append(seq)

    return prefill, decode
```

The three-way min is the whole idea:

```
chunk = min(remaining, budget, chunk_size)
```

- `remaining` — can't over-prefill the prompt.
- `budget` — can't blow `max_batched_tokens` minus the decodes.
- `chunk_size` — the dial you're tuning: how many tokens one sequence may add
  to a step.

**Finish in-flight prefills before admitting new ones.** Otherwise you accumulate
half-prefilled sequences that each hold KV memory while producing nothing,
memory pressure with no output, and every request's TTFT gets worse. The tests
encode this exact ordering (`test_in_flight_prefill_finishes_before_new_admission`).

**The boundary that bites.** The *last* chunk of a prefill produces the first
token. Once it runs, `num_prefilled == len(prompt_ids)` so `is_prefill_done`
turns true, and the sequence appears in `decode` on the *next* step. If the
runner treats "prefilled a chunk this step" and "decoding this step" as the same
thing, you get a duplicated or missing first token — the off-by-one
`test_final_chunk_transitions_to_decode` exists to catch.

The runner's half, for one sequence: feed the next `chunk_size_this_step` prompt
tokens to the model, let the KV cache grow by that many, and if the chunk was the
last one, the final position's logits are the first output token.

---

## Build it

1. Add `chunked_prefill` and `chunk_size` handling to `Scheduler.schedule()`.
2. Track `num_prefilled`; ensure `is_prefill_done` gates decode correctly.
3. `uv run pytest tests/test_11_chunked.py -v`
4. Measure on the workload built for it:

```bash
uv run python book/code/chunked_bench.py
```

`workloads.long_prefill` mixes 8 long prompts (~3000 tokens) with 24 short ones.
**Report p99 latency of the SHORT requests specifically**: that's where the
damage was, and `Request.tag` marks them.

**Predict first:** what happens to mean latency? To p99? To throughput?

---

## What you should see

**p99 improves, substantially.** This is the headline.

**Mean barely moves.** Total work is unchanged, you redistributed it. If you only
watch the mean you'll conclude this lecture did nothing, which is exactly the
trap Lecture 04 warned about.

**Throughput roughly flat**, maybe slightly down from per-step overhead. If it
drops a lot, your chunk size is too small.

Sweep `chunk_size` over 128 / 512 / 2048 and plot p99 against throughput. You
should see the trade directly.

---

## Go deeper

- **[SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)**
  (Agrawal et al., 2023): the origin. "Piggybacking" is exactly the mechanism:
  prefill rides along in capacity decode wasn't using.
- **[Taming Throughput-Latency Tradeoff (Sarathi-Serve)](https://arxiv.org/abs/2403.02310)**
 : the follow-up, with stall-free scheduling.
- **Kiely §5.3.4** (p.141), long-context handling, where this matters most.
- **vLLM `vllm/v1/core/sched/scheduler.py`**: chunked prefill is on by default in
  V1. Look for the token-budget accounting.
- **Gordić, *Inside vLLM***, has a chunked-prefill section. Still hold off until
  Lecture 14, but note it exists.

---

## Check yourself

1. Why does chunked prefill cost so little throughput? *(Answer in terms of
   Lecture 01's bottlenecks.)*
2. Mean latency barely changed but p99 improved a lot. Explain both.
3. Chunk size 64: what improves, what gets worse, and why?
4. Why must in-flight prefills finish before new sequences are admitted?
5. A sequence gets a 400-token prefix cache hit on a 3000-token prompt, with
   chunk size 512. How many prefill steps does it need?

---

## Next

**[12. Speculative decoding](12-speculative-decoding.md)**: attack the
sequential nature of decode itself.

```bash
uv run python book/code/spec_bench.py
```

**Report acceptance rate alongside tok/s, always.** Without it you cannot tell a
real win from a lucky one, and the same code looks like magic on
`code_completion` and useless on `prose`.

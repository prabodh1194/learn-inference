# 13. CUDA graphs

**Build:** graph capture in the model runner · **Test:** `tests/test_13_graphs.py` (cuda)
**Moves:** per-step latency for small batches, sometimes a lot
**Prereq:** [12b. Structured output and adapters](12b-structured-output.md)

> **NVIDIA GPU required.** No CUDA-graph equivalent exists on MPS. Tests here are
> marked `cuda` and skip cleanly on a laptop.

---

## The problem

Every optimization so far assumed **the GPU** is the bottleneck, the chip that
does the math. For small batches during decode, it often isn't. The **CPU**, the
processor that runs your Python and organizes the work, is the one running out
of time instead. The two machines cooperate, and both must be paid.

One decode step is hundreds of small jobs. Each job is a **kernel**: a small
program the chip runs, one step of the model's math, like "multiply these two
matrices" or "add these two vectors". A single decode step launches hundreds of
kernels, one or more per layer, times 28 layers, plus the attention math
(scores, softmax, output), the MLP, the two normalizations, the RoPE step (the
rotation that tells the model where each token sits in the sequence), and
sampling at the end. For a typical engine the honest count comes to about 400:

```
400 kernels  ÷  28 layers  ≈  14 kernels per layer
```

Now the part that is easy to miss. Before the chip runs any of that work, the
CPU must tell the chip to run it, one call per kernel. Each call is **launch
overhead**: the fixed cost of telling the chip to run a program, roughly
**5–10 microseconds** of CPU time, spent on paperwork, not arithmetic. Draw
the step as a time line:

```
one decode step

CPU runs Python:   submit   submit   submit                   submit
(kernel calls)     kernel 1 kernel 2 kernel 3     ...         kernel 400
                     v        v        v                          v
GPU (the chip):   [kernel 1][gap][kernel 2][gap][kernel 3][gap]  [kernel 400]
                     ^ work,   ^ the chip has nothing to do: the next
                       a few µs   order is still being written by the CPU
```

The chip executes kernels back to back only while the CPU keeps new orders
coming. When the CPU falls behind, the chip sits idle between kernels, and that
idle time is real time you pay for. Do the arithmetic. Four hundred kernels at
7µs each:

```
400 × 7 µs  =  2,800 µs  =  2.8 ms        of pure launch overhead
```

If the decode step's actual compute is 3ms, then of every step:

```
2.8 ms / (2.8 ms + 3.0 ms)  =  2.8 / 5.8  =  0.48
```

you're spending **nearly half your time on submission**, just ordering the
work, before and between the times the GPU computes anything.

The signature is unmistakable once you know it: **the GPU has visible gaps in its
timeline, the step is slow, and making the GPU faster changes nothing.** You are
CPU-bound in a program that appears to be about GPUs.

> Read that as *gaps in the profiler timeline*, not "low utilization percentage."
> The coarse utilization metric is unreliable here for the reason Lecture 00
> flagged, it can read high during memory-bound decode regardless of useful
> work. What you want is Nsight Systems showing idle stretches between kernels
> (Lecture 15), which is a direct observation rather than a summary statistic.

??? question "But kernel launches are asynchronous: can't the CPU just run ahead and hide all this?"
    Launching is asynchronous, but it still costs the CPU its own time. The CPU
    submits a kernel and moves on to the next call without waiting for the chip
    to finish, so the CPU does get ahead, until it falls behind. Each call
    costs the CPU 5–10µs of its own; a small-batch kernel executes in a few
    microseconds. When the per-call CPU cost exceeds the per-kernel chip time,
    the queue runs dry: the chip finishes what it has and waits for the next
    order, which is exactly the gap Nsight shows between kernels. CUDA graphs
    remove the per-call cost, so the CPU stops falling behind in the first
    place.
    [Full answer](qa.md#but-kernel-launches-are-asynchronous-cant-the-cpu-just-run-ahead-and-hide-all-this)

---

## The idea

Decode steps are **identical in shape**, every single time. Same kernels, same
order, same tensor shapes, only the values in the tensors change. A **tensor**
is a grid of numbers, here the activations flowing through the model; its
shape is its dimensions. "Same shape, different values" means every step is
structurally the same work, only the numbers differ.

So record the launch sequence once and replay it:

```python
# capture (once)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# replay (every step)
static_input.copy_(new_tokens)   # write into the SAME memory
g.replay()                        # one launch instead of 400
```

Hundreds of individual launches collapse into a single graph launch. The GPU work
is unchanged; the CPU-side submission cost nearly vanishes.

### The constraint that shapes everything

> A captured graph replays **exactly** the same operations on **exactly** the same
> memory addresses.

Consequences, all of which you must design around:

**Static shapes.** A graph captured for batch 8 works only for batch 8. Engines
capture several graphs (1, 2, 4, 8, 16, 32…) and pad up to the nearest. That
padding is real waste, traded against launch savings.

**Static memory.** Inputs must be copied into the same buffers each time. You
cannot pass fresh tensors. A captured graph records addresses, not values: a
fresh tensor lives at a new address, and replaying the graph would silently
read whatever is in the old one. That is why the code above writes
`static_input.copy_(new_tokens)` before every replay, a copy into the recorded
slot.

> This constraint is why `StaticCache` exists (Lecture 05). A `DynamicCache`
> grows by concatenation, so its tensors move, new addresses every step, which a
> captured graph cannot follow. `StaticCache` pre-allocates to `max_cache_len` and
> writes in place, so the addresses hold still.
>
> The cost is the one Lecture 09 spent a whole lecture on: pre-allocating the
> worst case per sequence. **CUDA graphs and paged attention pull against each
> other**, and every engine resolves it by capturing graphs over the *block
> tables* rather than over contiguous cache tensors. Worth noticing in vLLM's
> `gpu_model_runner.py`.

**No data-dependent control flow.** `if token == eos: break` inside the captured
region doesn't work: the branch was fixed at capture time. Sampling and stopping
logic stay outside the graph.

**Prefill usually isn't captured.** Variable prompt lengths mean variable shapes.
Prefill is compute-bound anyway, so launch overhead is a much smaller fraction of
it: the payoff is concentrated in decode.

### `torch.compile` is the other half

Different mechanism, complementary result. Where CUDA graphs remove *launch*
overhead, `torch.compile` reduces the *number* of kernels through fusion, three
elementwise ops become one.

Fewer kernels also means fewer launches, so the two compound. Modern vLLM uses
both. Try them separately first so you can attribute the wins.

Watch for **recompilation**: a new input shape triggers a fresh compile, which is
slow. If your steady-state throughput is fine but latency spikes occasionally,
check whether shapes are varying.

---

## Build it

1. **Profile first.** Lecture 15 covers this properly, but a quick check now:

```python
with torch.profiler.profile(activities=[CPU, CUDA]) as prof:
    decode_step()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
```

**Large CPU time with low GPU time means launch-bound.** If you're not, this
lecture won't help you much, and knowing that is itself the point.

2. Capture a graph for your decode step at a fixed batch size.
3. Capture at several batch sizes; pad to the nearest.
4. `uv run pytest tests/test_13_graphs.py -v` on a CUDA box.
5. Measure at batch 1, 8, 32, 128, **the win should shrink as batch grows.**
6. Separately, try `torch.compile(model)` and measure it alone, then together.

---

## What you should see

**Large gains at small batch sizes.** At batch 1 you're maximally launch-bound.

**Diminishing gains as batch grows.** More compute per launch means overhead
matters less proportionally.

**Nothing for prefill.** Expected: it's compute-bound.

**Higher memory use.** Each captured graph holds its static buffers, the
reserved input, output, and scratch slots it replays into, one set per batch
size. Capturing many batch sizes costs real VRAM, which competes with your KV
cache.

That last point is worth sitting with: this optimization *takes* memory from
Lecture 09's budget. Every engine makes this trade, and it's a genuine trade
rather than a free win.

---

## Go deeper

- **[NVIDIA: Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)**
 : the mechanism, with launch-overhead measurements.
- **[PyTorch CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)**:
  `torch.cuda.graph` and the memory-pool constraints.
- **vLLM `vllm/v1/worker/gpu_model_runner.py`**: search for `capture`. Note the
  list of captured batch sizes and the padding logic.
- **nano-vllm `nanovllm/engine/model_runner.py`**: ~12KB, the readable version.
- **Kiely §4.1.3** (p.100), kernel fusion and reducing memory accesses.

---

## Check yourself

1. Your decode step takes 3ms with 400 kernel launches. What's the theoretical
   best case from graphs alone?
2. Why capture decode but not prefill?
3. Why can't `if token == eos: break` live inside a captured graph?
4. Graphs help at batch 1 and barely at batch 128. Why?
5. Capturing 8 batch sizes costs VRAM that would otherwise be KV cache. How would
   you decide the tradeoff?

---

## Next

**[14. Reading vLLM](14-reading-vllm.md)**: you've built it. Now read the real
thing.

**Nothing to implement.** This is the capstone: Gordić's *Inside vLLM*, then
nano-vllm file by file, then four targeted vLLM files, each diffed against what
you wrote.

Read it *now* and not earlier. Before you'd built a scheduler it would have
taught you vocabulary; here it teaches judgment.

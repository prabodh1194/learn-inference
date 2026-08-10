# Appendix A: Glossary

Terms as this book uses them, with the lecture that earns each one.

For a broader industry glossary, see Kiely, *Inference Engineering*, Appendix A
(p.209–230).

---

## The two phases

**Prefill**: processing the whole prompt in one forward pass, producing the first
token and filling the KV cache. **Compute-bound.** Determines TTFT. *(L01)*

**Decode**: generating one token per forward pass, each depending on the last.
**Memory-bound.** Determines TPS. *(L01)*

**TTFT**: time to first token. Dominated by prefill and queueing. *(L04)*

**TPOT / ITL**: time per output token / inter-token latency. Steady-state decode
speed, excluding the first token. *(L04)*

---

## The bottleneck vocabulary

**Arithmetic intensity**: operations per byte of memory traffic, for an
algorithm. *(L02)*

**ops:byte ratio**: peak FLOPS ÷ peak bandwidth, for a device. This is the
**ridge point** of the roofline. H100 ≈ 295, RTX 3090 ≈ 76. *(L02)*

**Memory-bound**: intensity below the ridge; the GPU waits on memory. Decode, at
~0.75 ops:byte. *(L01, L02)*

**Compute-bound**: intensity above the ridge; arithmetic units are the limit.
Prefill. *(L01, L02)*

**Roofline**: a chart that tells you whether an operation is limited by
arithmetic or by memory. Two ceilings: a **diagonal** (you can't compute faster
than memory feeds you; slope = bandwidth) rising into a **horizontal** (you
can't exceed the arithmetic units; height = peak FLOPS). Named for the shape.

The corner where they meet (the **ridge point**) sits at the ops:byte ratio.
Plot your algorithm's arithmetic intensity on the x-axis: left of the ridge is
memory-bound, right is compute-bound. Decode sits at 0.75 against an H100's 295.
*(L02)*

---

## Caching

**KV cache**: stored keys and values for previous tokens, so they aren't
recomputed. Valid because attention is causal. *(L05)*

**GQA**: grouped-query attention: fewer KV heads than query heads, shrinking the
cache proportionally. Qwen3-0.6B: 16 query, 8 KV. *(L05)*

**PagedAttention**: storing the KV cache in fixed-size blocks with a block table,
instead of contiguously. Virtual memory for the cache. *(L09)*

**Block**: the unit of KV allocation, typically 16 tokens. *(L09)*

**Block table**: a sequence's map from logical position to physical block. *(L09)*

**Internal fragmentation**: waste inside the last partial block. Bounded by
`block_size - 1` tokens, unlike contiguous allocation's unbounded waste. *(L09)*

**Prefix caching**: sharing physical blocks between sequences with identical
prefixes. *(L10)*

**Parent hash**: the previous block's hash, chained into this block's identity so
that identical tokens with different histories don't collide. *(L10)*

---

## Scheduling

**Static batching**: a fixed batch that runs until its longest member finishes.
Wastes slots on mixed-length traffic. *(L07)*

**Continuous batching**: scheduling per step, admitting and retiring sequences
mid-flight. Also called iteration-level scheduling. *(L08)*

**Scheduler / model runner**: the split that continuous batching forces: one
decides what runs, one executes a step. The architecture of every real engine.
*(L08)*

**Chunked prefill**: splitting a long prefill across steps, interleaved with
decode, to protect p99. *(L11)*

**Preemption**: evicting a running sequence under memory pressure, by swapping its
blocks to host memory or discarding and recomputing. *(L09)*

**Head-of-line blocking**: one large item delaying everything behind it. *(L11)*

---

## Speculation

**Speculative decoding**: drafting several tokens cheaply, then verifying them in
one forward pass. **Exact**, not approximate. *(L12)*

**Acceptance rate**: fraction of drafted tokens that survive verification. Report
this alongside tok/s or you can't interpret the result. *(L12)*

**Draft model / Medusa / EAGLE / n-gram**: the four ways to produce drafts, in
roughly increasing order of setup cost. *(L12)*

**Bonus token**: the extra token you get free when all drafts are accepted, from
the trailing logits. *(L12)*

---

## Kernels

**Kernel**: a function running on the GPU. *(L15)*

**Kernel fusion**: combining operations into one kernel so intermediates stay in
registers instead of crossing HBM. *(L16)*

**HBM vs. SRAM**: high-bandwidth memory (GB, slow) vs. on-chip memory (KB, ~100×
faster). The gap FlashAttention exploits. *(L17)*

**FlashAttention**: tiled attention with online softmax that never materializes
the N×N score matrix. O(N) memory instead of O(N²). *(L17)*

**Online softmax**: computing softmax incrementally with a running max and sum,
rescaling as new tiles arrive. *(L17)*

**Coalescing**: threads in a warp reading consecutive addresses so the hardware
merges them. Getting this wrong costs up to 32× bandwidth. *(L20)*

**Occupancy**: resident warps per SM. Higher hides latency; registers and shared
memory bound it. *(L20)*

**CUDA graph**: a recorded launch sequence replayed as one operation, removing
per-kernel launch overhead. *(L13)*

---

## Quantization

**W8A16 / W4A16**: weight-only quantization; compute stays FP16. The inference
default. *(L19)*

**Per-tensor / per-channel / per-group**: scale granularity, trading metadata
against accuracy. *(L19)*

**GPTQ / AWQ / SmoothQuant**: calibration methods. AWQ's premise: not all weights
matter equally. *(L19)*

**KV cache quantization**: compressing the cache rather than the weights. Buys
concurrency, not per-step speed. *(L19)*

---

## Parallelism

**Tensor parallelism (TP)**: splitting tensors within each layer across GPUs. Two
all-reduces per layer. A **latency** optimization. *(L22)*

**Pipeline parallelism (PP)**: splitting layers across GPUs. Bubbles; poor
latency. Multi-node. *(L22)*

**Expert parallelism (EP)**: whole MoE experts on different GPUs. All-to-all token
routing. A **throughput** optimization. *(L23)*

**Column/row parallel**: the complementary sharding that lets an MLP need only one
all-reduce. *(L22)*

**All-reduce / all-to-all**: the collectives TP and EP respectively depend on.
*(L22, L23)*

**MoE**: mixture of experts: many MLPs, few active per token. *(L23)*

**Total vs. active parameters**: what you must store vs. what runs per token.
DeepSeek-V3 is 671B total / 37B active, and needs VRAM for 671B. *(L23)*

---

## Production

**Open vs. closed loop**: load arriving at a fixed rate vs. clients waiting for
responses. Closed-loop tests cannot show overload. *(L25)*

**The knee**: the offered load past which throughput stops rising and p99 runs
away. Your real capacity. *(L25)*

**Cache-aware routing**: routing to the replica that already holds the prefix,
rather than round-robin. *(L27)*

**Disaggregation**: running prefill and decode on separate workers, transferring
the KV cache between them. *(L27)*

**Cold start**: time from scale-up to serving traffic. Minutes, not seconds.
*(L28)*

**Fleet utilization**: fraction of paid GPU-hours doing paid work. Distinct from
GPU busy-percentage, which is a poor scaling signal. *(L28)*

**Cost per million tokens**: the unifying metric. Dominated by **fleet
utilization** at low load, not by engine quality. *(L28)*

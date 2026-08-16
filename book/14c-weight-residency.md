# 14c. Weight residency: the memory curve is a policy, not a fact

> Lecture 14b's engine map is one fact away from being useful: the model is
> **37 GiB** and the machines it runs on have **64–128 GiB of unified memory**.
> The weights are not in memory or out of memory — they are *resident to a
> degree you choose*, and each choice has a measured cost. This lecture is
> about how h3.c chooses, and why "the weights are loaded" is not one thing.

## Three policies, one checkpoint

The checkpoint is a pile of BF16 safetensors: the DiT and both encoders, about
37 GiB total. How do they get from disk into the GPU?

1. **mmap, zero-copy** — the file is mapped, Metal binds the pages lazily, the
   GPU faults them in. Resident usage ≈ pages actually touched. Fastest *start*
   on machines where GPU can read it, but the GPU pays the page-fault price
   inside kernel time, and residency is page-granular.
2. **Copy into allocated buffers** — explicit `memcpy` at load. Costs the copy
   once, at startup; residency afterwards is what you asked for, and GPU
   access is direct.
3. **Double-buffered streaming** — the model is split into two slices; while
   the GPU computes on slice A, a CPU thread reads slice B from disk into
   memory (14c's third section). Residency drops to *one slice*: ~2 GiB.

The README reports a nice inversion (`h3.c:538`): on the M5 Max, **mmap
zero-copy beats the explicit copy**; on the M3 Max, **the explicit copy beats
mmap**. Same code, same checkpoint, opposite ranking.

??? question "Why would copying weights beat memory-mapping them?"
    mmap is not "free" — the pages are faulted in on first touch, and on a
    unified-memory GPU that first touch can happen *inside* a Metal kernel,
    with the fault paid as GPU time and a page-granular residency you can't
    reason about. If the copy is cheaper than the faults and the copy makes
    GPU access direct, copying wins. The M5 has enough bandwidth and the right
    page behavior that faults stay cheap — so the ranking flips.
    [Full answer](qa.md#why-would-copying-weights-beat-memory-mapping-them)

The lesson is not "use copy on M3": it's that **residency is a policy with
three implementations and the measurements decide, per machine**. The code
keeps both paths and picks by device name. That is the same discipline Lecture
26 preaches at the serving level, one level down: don't argue about the
memory system, measure it.

## Streaming: 37 GiB model, 2 GiB resident

The flagship residency trick is `h3_streamed_forward` (README: 4.2). The 50
DiT blocks are split into two slices along the layer axis — for the 864-class
geometry, layers 1–25 in one half, 26–50 in the other. The forward pass runs
the first half, and while the GPU is busy, a CPU worker reads the *second*
half's weights from SSD into memory. Then the pass runs the second half, which
is already resident. The SSD reads and the GPU math overlap, and the resident
set shrinks to one slice plus the persistent parts:

```
full model:     50 × ~930 MiB   ≈  46.9 GiB
resident slice: 25 × ~930 MiB   ≈  23.4 GiB   ← still big
```

…which is why the README's headline numbers use a *coarser* split at the
encoder level. In the reported configuration the resident set goes from
36.5 GiB to **2.0 GiB** — an 18× cut — at a measured cost of 84% slower
forward time at 512-class (1.35 s → 2.49 s) and 26% slower at 864-class
(2.14 s → 2.68 s). The numbers, quoted: *"streaming from the SSD takes 2.49
seconds vs 1.35 seconds, at 512-class... using the SSD storage is 84% slower,
at 864-class only 26% slower."*

The math that makes it work: the SSD reads at 13–14.6 GiB/s, so half a slice
arrives while the GPU is still grinding on the other half. The reason 864
suffers less is that its steps are so much heavier that the read time hides
under compute time. The reason it's ever acceptable is the same as Lecture
04's measurement discipline: **you can quote the exact price of the memory
you're saving**, in seconds per generation, and decide whether 18× of RAM is
worth it.

There is a warm-up subtlety worth copying: the first block of the second slice
is prefetched *immediately* at load, so interactive use isn't greeted by a
first-step stall — a fix for the classic "streaming feels slow" complaint,
which is really "the first touch was slow."

## The prefetch ring: encode without stalling

The text/vision encoder path has the same problem at a different scale. The
Qwen3-VL encoder is 46.9 GiB of weights (52 layers for the 27B configuration),
and it's needed once per generation, then freed. Loading it with a single
blocking read would serialize the whole pipeline on disk. Instead, the encoder
weights stream through a **prefetch ring**: 8 worker threads, ring depth 2–3,
blocks of ~930 MiB each. While the GPU encodes the prompt with block *n*, the
workers are fetching blocks *n+1* and *n+2* from disk.

The residency arithmetic, derived: a depth-2 ring holds `depth + 1` blocks in
flight at once (one being consumed by the GPU, `depth` being fetched).

```
ring residency @ depth 2  =  3 × 930 MiB  ≈  2.72 GiB
ring residency @ depth 3  =  4 × 930 MiB  ≈  3.63 GiB
```

At 13–14.6 GiB/s, a 930 MiB block arrives in 65 ms, and a depth-2 ring keeps
the GPU's consumption covered: by the time the GPU wants block *n+2*, it has
been in flight for the duration of blocks *n* and *n+1*. The tradeoff is
explicit: **more depth = more residency = more slack**, and both are measured
in the same units (MiB and ms). This is Lecture 07's producer/consumer
question, but the producer is the disk.

## The re-read decision

Not everything is streamed. The video decoder VAE (9.7 GiB) is loaded into
memory once and stays for the decode phase — even though it's only *read*,
never written. Why not stream the decoder too? Because the VAE's weights are
consumed by short, bandwidth-hungry convs, the decode phase re-reads chunks
per tile, and the model is needed for the entire video decode at once: every
tile touches most weights. Re-reading 9 GiB per tile (or even per video) is
pricier than holding it. The rule of thumb, stated as a question: **is the
working set touched repeatedly with short, bandwidth-hungry kernels, or
traversed once per generation with room to overlap the disk under compute?**
Resident for the first, streamed for the second.

## Check yourself

1. Streaming halves the DiT slices, yet the README reports 36.5 → 2.0 GiB, not
   36.5 → ~18 GiB. What must be true about the *reported* split for that
   number to work?
2. Why does 864-class streaming cost only 26% while 512-class costs 84%?
3. The prefetch ring and the SSD streaming are the same idea at two scales.
   What differs about the *failure mode* when the disk is too slow in each?

## Next

**[14d. Fusion: launch and memory trades](14d-fusion.md)**: the DiT fuses
kernels — and the point of fusing is usually not what you'd guess.
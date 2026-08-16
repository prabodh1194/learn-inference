# 14b. Reading h3.c — the engine map

> Part II taught you to build engines. This series reads a second real one, in
> the open: antirez's **h3.c**, a native Metal inference engine for MiniMax H3 —
> a video+audio diffusion model — written in C for Apple Silicon. It's ~1.3 MB
> of C, Objective-C, and Metal shaders, MIT-licensed, one binary, no Python.
> It is also the only production-grade engine in this book that runs on the
> hardware you actually own.

Why should a video-generation engine live inside a book about LLM serving?
Because every lesson of Part II shows up again here, wearing different
clothes:

- **Decode is memory-bound** → a 37 GiB checkpoint on unified memory (one shared pool the CPU and GPU both address — no host/device copy) makes the
  *weight residency* question the whole ballgame (14c).
- **Launch overhead hurts** → the DiT (diffusion transformer) fuses kernels to kill dispatches — one kernel launch on the GPU's command queue (14d).
- **Memory is a budget** → activations are aliased by lifetime, not bought (14e).
- **CPU/GPU are async** → command buffers (the GPU's work queues, filled by the CPU) are split so encoding hides under
  execution (14f).
- **Quantization is a trade** → int8 lands with SSIM receipts (14g).
- **Optimization changes outputs** → the aggressive paths say so, out loud (14h).

One honesty rule for the whole series: **every speed number below is antirez's
own measurement on an M3/M5 Max, quoted from the README** — labeled as
upstream, never as something we reproduced. The one thing verified locally is
at the end of this lecture.

## The machine

H3 is a **33B-parameter diffusion transformer** that generates video *and*
audio from a text prompt. One "token" of output isn't a word — it's a patch of
pixels (latent channels — the model's internal feature planes; here 24 of them, patched 2×2×2 into width-96 rows) plus a width-32
audio row. A 512×512, 22-frame video is a grid of about 2,800 such rows, and
generating it is a **denoising loop** (each step removes a little more of the
noise the latent started as): ~20 steps, each step a full 50-block
forward pass over the whole grid, predicting a velocity (the direction the
latent moves toward a clean frame).

That loop is the analogue of your decode loop. Each denoise step is a full
model forward — same weights, same shapes, same layout, step after step. The
consequence is the same as in LLM decode, one level up: **the per-step
overhead is paid 20 times**, so it's worth optimizing per-step costs (14d,
14e, 14f) and even worth *skipping* steps (14h).

The pipeline has three phases, and this is the first lesson:

```
encode  →  DiT (denoise loop)  →  decode
tokenizer         50 blocks         video VAE (36 blocks)
Qwen3-VL text                      audio VAE (BigVGAN)
Qwen3-VL vision                    ffmpeg mux
video/audio VAE encoders
```

Each phase loads its weights when it starts and frees them when it ends. The
README states the rule outright: the transformer, the Qwen encoder, and the
decoders **"never have to coexist in unified memory."** On a 128 GB M5 Max the
end-to-end peak is ~40 GB with zero swaps — about a third of the machine,
which is the point: the phases each fit in the budget
*separately*, and the scheduler is what makes that true. This is Lecture 09's
budget discipline, applied at the component level instead of the token level:
**peak memory is a scheduling problem, not a purchase.**

## The one binary, three front-ends

The objective here is to see how one binary serves three use patterns from a
single model state — the same CLI question your own engine answers, with a
memory axis added.

`main.c` is a thin dispatcher over three modes:

- **One-shot**: `-p PROMPT ...` → build params, call `h3_generate`, exit.
- **Interactive**: no `-p` → a linenoise REPL (`!seed random`, `!again`,
  `!cache`, `!first`, `!ref-image`…), one model context reused across every
  generation in the session.
- **Inspection**: `--info` → device + checkpoint inventory, **without mapping
  any weights**.

That last mode is quietly profound: the model metadata lives in safetensors (a tensor-serialization format with a header index separate from the payload bytes)
headers, and the loader reads *every header once* and never touches payloads
(`h3_weight_store_open`). A 37 GiB model is fully inspectable through an
index pass that costs kilobytes. Same instinct as your engine's `--info`: the
index is cheap, the payload is not — so never pay for the payload to answer a
question the index can answer.

## The session cache: cache the prefill, re-roll the sampling

The objective here is to make repeated generations cheap: cache everything
that doesn't depend on the seed, and re-roll only what does.

The interactive session keeps three things resident, each keyed by its true
dependencies (`h3.c:139-192`):

| Cache | Key | What it holds |
|---|---|---|
| conditioning | `mode\|prompt\|render\|frames\|media size:mtime` | exact BF16 (brain float16: 8-bit exponent, 7-bit mantissa) text embedding + condition rows |
| prepared DiT | conditioning key + shape + steps + layers + quant flags | the loaded model, reset per run |
| video decoder | VAE (variational autoencoder — the decoder that turns the latent into frames) path + latent geometry | resident weights |

The seed is deliberately **absent** from every key. "Same prompt, new seed"
re-encodes nothing, re-loads nothing — it re-rolls the Gaussian noise and
re-runs the loop. This is exactly the book's prefix-cache lesson (Lecture 10)
in a second model family: the expensive prefix (the prompt → embedding pass,
which is *the prefill*) is cached bit-exactly — the cache stores the actual
BF16 values the encoder would produce, so a hit is not an approximation.

Two disciplines make the cache correct rather than merely useful:

1. **Keys encode everything that changes the output and nothing that doesn't.**
   Media files are keyed by `size:mtime`, so editing an anchor image
   invalidates the conditioning with no manual flush. Steps/layers/quant flags
   invalidate the DiT. The seed and the reuse factor don't — and don't need to.
2. **A failed denoise run poisons the cache.** If a generation fails while the
   DiT is cached, the cached artifact is dropped rather than retried — a
   possibly-corrupt artifact is never kept.

## What I verified locally

`make test` on the M1 (8 GB, no checkpoint, no Xcode toolchain — the Metal
kernels compile at runtime, exactly as designed):

- **1,768 host checks pass** — the deterministic CPU suite (frame-shape law,
  schedules, layout builder, RNG, samplers).
- **Metal primitives match host references** — the audio VAE's GPU kernels
  (Conv1d, ConvTranspose1d, SnakeBeta, weighted norms) agree with their CPU
  implementations to ~6e-8 max absolute error. The Metal path actually
  executes on this machine.
- Everything else skips: the released weights/fixtures aren't installed, and
  FFmpeg isn't on PATH.

So the engine builds clean and its GPU core runs on a first-gen Apple Silicon
Mac. The M5-only paths (TensorOps — Apple's low-precision matrix-multiply path — int8, zero-copy weights) are runtime-gated
and fall back gracefully — a design choice worth noting: **capability
detection by device-name string**, `[device.name rangeOfString:@"M5"]`, with
env-var overrides for everything. It's fragile as hardware detection goes, and
it's *deliberate*: antirez measures on the machines he has and makes the
choice explicit rather than pretending to query a feature API that doesn't
answer the real question.

## Check yourself

1. Why is a denoising loop (not a one-shot forward) the right place for the
   per-step optimizations in 14d–14f?
2. The conditioning cache key excludes the seed. Why is that safe, and what
   would break if it also excluded the prompt?
3. Why can `--info` work without mapping any weights?

## Next

**[14c. Weight residency](14c-weight-residency.md)**: a 37 GiB model on unified
memory — map it, copy it, or stream it, and measure all three.
# 19c. GGUF — the quantized model file format

**Build:** nothing · **Test:** none · **Moves:** reading/understanding the format every local engine reads
**Prereq:** [19. Quantization](19-quantization.md), [19b. FP8](19b-fp8.md), [14g. Quantization with receipts](14g-quantization.md)

---

## The problem

Lectures 19/19b covered *how* to quantize. This one covers **where quantized
weights live**: a model file format. GGUF (GPT-Generated Unified Format,
Georgi Gerganov's llama.cpp family, ~Aug 2023) is the de-facto distribution
format for quantized LLMs — every local engine reads it (llama.cpp, llama.rn,
**ds4**, Ollama, LM Studio). If you meet local inference, you meet GGUF.

GGML, the name you'll also see, is just the older C tensor library that
llama.cpp grew out of — the *format* it defined was replaced by GGUF (better
metadata, extensible, fixed endianness). History footnote, not a topic: **GGUF
is the format, GGML is the dead library.**

## The idea

A GGUF file is a binary container: **metadata KV store + tensor payload**,
all little-endian, laid out for **mmap** (the engine memory-maps the file and
reads weights straight off disk — no load phase; see [14c. Weight
residency](14c-weight-residency.md) for why that matters on 8GB machines).

```
   GGUF layout
   ├─ magic "GGUF" + version
   ├─ metadata KV: architecture, n_layers, vocab, chat template (Jinja!),
   │                tokenizer config — everything a loader needs
   ├─ tensor info table: name, shape, type, offset
   └─ tensor data, contiguous, mmap-friendly
```

The tensor **type** encodes the quantization — and this is the interesting
part: quantized tensors are stored *block-wise*, scale included:

```
   Q4_0: 32 weights → 1 fp16 scale + 32 int4 (packed 16 bytes)
   Q4_K / Q6_K: super-blocks (256 weights, finer scales + offsets)
   IQ2/IQ3/IQ4: trellis-coded — quantize *groups of values together*
               (adds ~0.25-1 bit of effective precision per weight)
```

That's Lecture 19's `scale × (w - zero_point)` per *group*, baked into the
file format itself — same math, different container. **The scale is in the
file**, not computed at load.

## Why the format choice is a system decision

| | GGUF block quant (Q4_K) | FP8 (19b) |
|---|---|---|
| hardware | any (dequant in software, CPU-friendly) | tensor cores only (Ada+) |
| distribution | file ships quantized | needs calibrated scales + conversion |
| precision | good, block-adaptive | better, exponent-native |
| the catch | CPU/edge path, llama.cpp family | GPU-only, engine must support |

One is a **file format for the ecosystem** (any device, weights pre-quantized
at conversion time); the other is a **hardware feature** (per-GPU). A server
with FP8 tensor cores runs FP8; a laptop or an 8GB M1 runs a GGUF Q4 file.
This is the same hardware-format split the [field notes](field-notes.md)
record: format choice follows the machine, and the eval decides what's
acceptable (Lecture 19's rule).

## Where to look in real engines

- **ds4** (antirez, single-file C engine for DeepSeek 4 Flash/PRO): loads
  GGUF, keeps Q4_K/IQ2-class files resident on Metal/CUDA — the file layout
  IS the memory layout: `gguf` reader walks the tensor table straight into
  GPU buffers. Read its `gguf.c` after [14b. Reading h3.c](14b-reading-h3.md)
  for a second author's take on the same problems.
- **llama.cpp**: `ggml-quants.c` — the block dequant kernels (Q4_K etc.).
- **Ollama / LM Studio**: GUIs over the same format; `ollama show --modelfile`
  reveals the KV metadata.

## Go deeper

- **GGUF spec** (`ggml/src/gguf.cpp` in llama.cpp, or the format.md in the
  repo) — the KV metadata table is the spec's soul.
- **ds4 repo** (github.com/antirez/ds4): GGUF reading + inference in one
  file; h3.c comparisons in [14b–14h](14b-reading-h3.md).
- **Kiely §5.1.3**, block quantization's accuracy rationale.

## Check yourself

1. Why is little-endian + mmap the right layout for a model file?
2. Q4_K stores 256 weights with scales; FP8 stores per-tensor scales. When
   does the block version win on GPU?
3. Why does the chat template live in the GGUF metadata? (Tie to
   troubleshooting I2.)
4. A Q4_K file is 4.3 GB for a 7B model. Where did the extra 1.3 GB go?
5. ds4 keeps IQ2 files resident on an 8GB GPU. What does that choice cost
   (Lecture 14c's terms)?

## Next

Back to the main line: **[20. Raw CUDA](20-raw-cuda.md)**, or
**[21. JAX and XLA](21-jax-and-xla.md)** if you're continuing the thread.
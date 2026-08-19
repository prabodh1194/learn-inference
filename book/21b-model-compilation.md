# 21b. Model compilation — TensorRT and torch.compile

**Build:** nothing · **Test:** none · **Moves:** how production serving engines turn models into kernels
**Prereq:** [21. JAX and XLA](21-jax-and-xla.md), [13. CUDA graphs](13-cuda-graphs.md), [14d. Fusion](14d-fusion.md)

---

## The problem

Everything in this book so far launches kernels one at a time. Eager PyTorch
costs a launch per op — Lecture 13 counted the price (~10µs × hundreds of
ops). CUDA graphs (Lecture 13) eliminate the *replay* cost but not the
*fusion* gap: ops still run as separate kernels. FlashAttention (Lecture 17)
and your Triton kernels (16–18) fuse by hand, op by op.

The industry's other answer: **compile the whole model into fused kernels
once, at build time.** Two flavours dominate production:

- **torch.compile** — PyTorch's JIT: trace your Python, fuse it, emit Triton.
- **TensorRT-LLM** — NVIDIA's serving engine: take a network definition,
  autotune fused kernels, ship an *engine file*.

Both are the XLA idea from Lecture 21 applied to serving instead of training.

---

## The idea

```
   eager        one launch per op                (you've measured this, L13)
   CUDA graphs  replay a recorded launch list    (L13 — no Python, no fusion)
   torch.compile  Dynamo: trace Python → FX graph; Inductor: fuse + emit Triton
   TensorRT-LLM   network def → autotuned engine (fused kernels, static shapes)
```

Compilation wins where CUDA graphs stop: **fusion**. 14d's launch-cost theory
is what the compiler automates. `bias + GELU + residual` (3 launches, 3
weight-reads) becomes one kernel: same math as your hand-fused Triton, found
by a compiler instead of by hand.

| | CUDA graphs | torch.compile | TensorRT-LLM |
|---|---|---|---|
| capture | launch list | Python trace | network def |
| fusion | none | Inductor | TRT kernel autotuning |
| shapes | fixed | static at trace | compiled per config |
| built for | eager engines | training/serving | serving |
| output | replay | Triton code | engine file |

## The one concept that transfers: graph breaks

Dynamo traces Python by executing it; anything it cannot trace becomes a
**graph break** — a boundary where execution falls back to eager, restarting
the Python overhead you compiled to remove. The break list is Lecture 13's
constraint list wearing a compiler's clothes:

```
   graph breaks ←── data-dependent control flow (I11's checklist, L13)
   ├─ boolean indexing / .item() / if tensor > 0
   ├─ torch.nonzero
   ├─ dynamic shapes (lengths known only at runtime)
   └─ unsupported ops
```

Each break = one more eager region = the launch cost you thought you removed.
The debugging habit is the same as troubleshooting I11: **rewrite control flow
as arithmetic**, or mark shapes static. `TORCH_LOGS=graph_breaks` lists them;
`TORCH_LOGS=output_code` shows the emitted kernels — read the fusion the way
you read your own Triton.

## Where each engine sits

- **vLLM (V1):** eager + CUDA graphs (L13). No compile step — the fallback
  story from I7 (`flash_attn` wheels) and I6 (FlashInfer JIT) exists *because*
  kernels come from wheels, not compilation.
- **SGLang:** eager + default CUDA graphs (26b) — same family.
- **TensorRT-LLM:** compilation-first. Kernels come from the engine file, not
  the wheel ecosystem — which is why TRT servers are immune to the I7/I6
  failure classes entirely (and pay for it in build time and shape rigidity).
- **XLA (L21):** the same idea for training/TPU — HLO → fused kernels.

Production insight, from the troubleshooting page: engines are either
**eager + graph replay** (vLLM/SGLang) or **compiled** (TRT-LLM). The
difference decides which incidents you can have. Choose per workload: compile
time + static shapes vs wheel fragility + dynamic flexibility.

## Interview / practitioner pointers

- **Allen Philip J** — "TensorRT: From Frustration to Production" and
  "Building Custom TensorRT Plugins" (allenphilip93.github.io): a working
  engineer's account of fusion and custom kernels inside TRT engines.
- Troubleshooting **I8**: Ray's managed layer wraps a compiled path — the
  version deadlock lived in the *build-time* dependency graph, which is why
  an HTTP boundary fixed it.

## Go deeper

- **[torch.compile docs](https://pytorch.org/docs/stable/torch.compiler.html)**:
  Dynamo/Inductor design, graph-break FAQ.
- **[TensorRT-LLM docs](https://github.com/NVIDIA/TensorRT-LLM)** — build an
  engine, inspect `trtexec` timing.
- **Kiely §4.2.3** — compilers in the inference stack (same pointer as L21).

## Check yourself

1. CUDA graphs remove launch *overhead*; compilation removes launch *count*.
   Draw the pipeline for a 5-op attention block under each.
2. A model with heavy data-dependent control flow compiles to many graph
   breaks. What does `TORCH_LOGS=graph_breaks` show, and what's the fix?
3. Why is TRT-LLM immune to I7's `flash_attn` wheel problem? What does it
   pay instead?
4. Which of I11's un-capturable ops is also a graph break? Name two that
   aren't.
5. When would you pick eager+CUDA-graphs over a compiled engine for serving?

## Next

**[22. Tensor parallelism](22-tensor-parallelism.md)** — collectives, the
thing compilers insert and you'll write by hand. (Or [19b. FP8](19b-fp8.md)
if you came from the quantization branch.)
# 25b. Deployment environments

**Build:** nothing · **Test:** none · **Moves:** your ability to deploy without surprise
**Prereq:** [24. Serving](24-serving.md), [25. Load testing](25-load-testing.md)

---

## The problem

You built the engine. It works on your machine. You containerise it, push to the
cluster, and it fails — `import vllm` not found, FlashInfer can't find
`cublasLt.h`, the resolver deadlocks on `transformers<5` vs `transformers==5.10.2`.

The engine is correct. The **environment** is the bug.

Ten of the fourteen incidents in [Production troubleshooting](troubleshooting.md)
are environment failures: ignored config, wrong interpreter, runtime image
missing dev headers, version graphs with no solution. The lectures teach the
engine's internals; this lecture teaches the soil it runs in.

---

## The idea

**"Installed" is not a property of the machine. It is a property of one
interpreter.**

A Python package lives in `site-packages` of a specific interpreter. `uv add`
writes to `.venv/lib/python3.x/site-packages/`. The cluster job runner invokes
`/opt/conda/bin/python`. Two interpreters, two `site-packages`, same disk,
different answers. `ModuleNotFoundError` means "this interpreter hasn't seen
it" — not "it's not on disk."

**Containers encode assumptions.** A `-runtime` image assumes compilation
already happened. FlashInfer JIT-compiles at *runtime*, per TP shape. Changing
TP invalidates the cache → compile triggered → `cublasLt.h` not found. The
image was wrong for the workload.

**Version constraints compound.** Ray pins vLLM, vLLM pins transformers, your
model pins transformers. Four layers → empty intersection. No resolver cleverness
fixes a contradiction.

**Process boundaries are dependency boundaries.** SGLang spawns a model runner
subprocess. It doesn't inherit `PYTHONPATH`. Your custom model imports in your
process, not in the runner's. The fix isn't fighting the boundary — it's
putting code where the child looks (patch installed source) or crossing the
boundary cleanly (HTTP).

---

## The method

### 1. What "installed" actually means

```
interpreter → site-packages/ → package
```

- `sys.executable` tells you which interpreter is running.
- `sys.path` tells you where it looks.
- `pip show -f package` shows where files *actually* landed.
- `uv run python` activates the venv for that command only.

**The habit:** when "it's installed" fails, print `sys.executable, sys.path`
from *inside the failing process*. Not from your terminal.

### 2. Runtime vs devel images — the JIT trap

| Image type | Contains | Assumption |
|------------|----------|------------|
| `-runtime` | CUDA driver, libcudart, libcublas | compilation already done |
| `-devel` | runtime + `nvcc`, headers, `nvrtc`, `cublasLt.h` | you might compile |

FlashInfer, `flash-attn`, custom CUDA ops: they compile at runtime. On a
`-runtime` image, they work until cache invalidation (TP change, new GPU,
new shape), then fail with header errors.

**Fix options:**
- Use `-devel` base image, or `apt-get install cuda-nvrtc-dev-12-8 libcublas-dev-12-8 ninja-build`
- Lock shapes so cache stays valid (TP=4 fixed)
- Pre-compile and ship wheels (harder, but eliminates runtime compile)

### 3. Subprocess isolation in serving engines

vLLM and SGLang run the model in a separate process (Lecture 24). The child:
- May use `spawn` (fresh interpreter, no `PYTHONPATH`)
- May not inherit env vars
- Has its own `sys.path`

Your custom model must be importable *by the child*. Options:
1. **Patch installed source** (edit `.venv/.../sglang/srt/models/qwen3.py`) — ugly, reproducible if scripted, survives upgrades if re-applied
2. **HTTP boundary** — serve custom model from separate process, call via API (clean, independent versions)
3. **Bake into image** — `COPY` your model code into the Docker image at build time

### 4. Version graphs — recognise impossibility

Write the constraint chain by hand:

```
Ray 2.55.x     →  vllm >= 0.18
vllm >= 0.18   →  transformers < 5
Your model     →  transformers == 5.10.2
                      ───────────────────
                      transformers < 5 AND == 5.10.2
                      NO SOLUTION
```

If contradictory: **decouple**. HTTP boundary = dependency boundary. Each side
resolves its own graph.

### 5. Wheel compatibility on new hardware

New GPU → new arch → prebuilt wheels lag. `torch.cuda.get_device_capability()`
tells you the arch (e.g., 9.0 for H100, 10.0 for B200). Check the wheel's
`--platform` tag before debugging upstream.

---

## Build it

No code. This lecture is a checklist you run *before* deploying:

```bash
# 1. Which interpreter?
python -c "import sys; print(sys.executable)"

# 2. Does the failing process see the package?
# (run INSIDE the failing job/container/subprocess)
python -c "import sys; print(sys.executable, sys.path)"

# 3. Container: runtime or devel?
# Dockerfile: FROM nvidia/cuda:12.8-devel-ubuntu22.04
# or: RUN apt-get update && apt-get install -y cuda-nvrtc-dev-12-8 libcublas-dev-12-8 ninja-build

# 4. Version constraints — write them out
pipdeptree  # or uv tree
# If contradictory → add HTTP boundary

# 5. Wheel arch match?
python -c "import torch; print(torch.cuda.get_device_capability())"
# Compare to wheel's platform tag (e.g., cp310-cp310-manylinux_2_28_x86_64)

# 6. Subprocess import test
# In parent: spawn child that prints sys.path and tries import
```

---

## What you should see

- `sys.executable` matches the venv you expect
- No `ModuleNotFoundError` for packages `uv add` installed
- No header errors (`cublasLt.h`, `nvrtc.h`) at runtime
- Resolver succeeds, or you have a documented HTTP boundary decoupling the graphs
- Custom model loads in engine's model runner (patched or HTTP)
- Wheel arch matches `torch.cuda.get_device_capability()`

---

## Go deeper

- [Production troubleshooting](troubleshooting.md) — the 14 incidents this lecture explains
- `man 7 environ` — environment variable mechanics
- `man 1 ld.so` — dynamic linker, `LD_LIBRARY_PATH`, `RPATH`
- PEP 425 — platform tags for wheels
- Docker multi-stage builds — compile in devel stage, copy artifacts to runtime stage

---

## Check yourself

1. Your Dockerfile uses `cuda:12.8-runtime`. You add FlashInfer. What breaks and when?
2. `uv add vllm` works. Cluster job fails `import vllm`. One command to prove why.
3. Ray pins `vllm>=0.18`, your model needs `transformers==5.10.2`, vLLM 0.18 needs `transformers<5`. What do you do?
4. SGLang subprocess can't import your custom model. Two fixes — one ugly/fast, one clean/slow.
5. New GPU (capability 10.0). `flash-attn` wheel is `cp310-cp310-manylinux_2_28_x86_64`. Will it work?

---

## Next

[26. Versus vLLM](26-versus-vllm.md) — fair benchmarking methodology, now that you can deploy both.
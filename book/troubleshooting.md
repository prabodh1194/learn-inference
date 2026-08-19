# Production troubleshooting

Fourteen real failures from running **vLLM** and **SGLang** in production on
B200 hardware — what broke, what it actually was, and why.

The book's lectures explain how inference engines are *supposed* to work. This
page is what happened when they didn't. Most of these cost hours, and almost
none of them were the bug they first appeared to be.

**They are organised by root cause, not by symptom.** That reorganisation is
the point. Fourteen incidents is a list; seven failure modes is a diagnosis,
and the second one transfers to bugs you have not met yet.

!!! tip "How to read this"
    Stuck right now? Use the **symptom index** below.

    Trying to get better at being stuck? Read the patterns in order. Each one
    ends with *how to recognise it next time*, which is the part worth
    remembering.

---

## Incident map

Each incident has an ID so the indexes below can point at the same thing from
multiple angles.

| ID | Incident | Engine | Pattern |
|----|----------|--------|---------|
| I1 | `VLLM_ATTENTION_BACKEND` ignored by V1 engine | vLLM | 1 |
| I2 | `enable_thinking` is cosmetic on Qwen3 | SGLang | 1 |
| I3 | `SGLANG_EXTERNAL_MODEL_DIR` never existed; registry subprocess gap | SGLang | 1 |
| I4 | `import vllm` fails inside cluster job scripts | vLLM | 2 |
| I5 | SGLang subprocess cannot import custom model | SGLang | 2 |
| I6 | FlashInfer JIT cannot find CUDA headers | vLLM | 3 |
| I7 | `flash_attn.ops` fails to import on new GPU generation | vLLM | 3 |
| I8 | Ray's LLM serving layer vs. the model you need | vLLM | 4 |
| I9 | RL trainer's vLLM integration segfaults | vLLM | 4 |
| I10 | Ray submit hangs in `PROVISIONING` (namespace not enabled) | platform | 4 |
| I11 | Indexed assignment breaks CUDA graph capture | custom | 5 |
| I12 | TP=2 crashes during graph capture on B200 | SGLang | 5 |
| I13 | `--max-model-len` too small: HTTP 400 on long documents | vLLM | 6 |
| I14 | `max_new_tokens` truncation manufactures a quality failure | eval | 6 |
| I15 | `loss = 0`, `grad_norm = nan` on fresh classification head | training | 7 |

---

## Symptom index

| What you see | Pattern |
|---|---|
| You set an env var / kwarg; behaviour is unchanged, no error | [1 — the knob that does nothing](#pattern-1-the-knob-that-does-nothing) |
| `ModuleNotFoundError` for a package you definitely installed | [2 — which Python is "installed"?](#pattern-2-which-python-is-installed) |
| `cublasLt.h` / `nvrtc.h` / `curand.h` not found, at *runtime* | [3 — the container is not the machine](#pattern-3-the-container-is-not-the-machine) |
| `import flash_attn.ops` fails on a new GPU generation | [3 — the container is not the machine](#pattern-3-the-container-is-not-the-machine) |
| A resolver refuses to install anything at all | [4 — the version graph has no solution](#pattern-4-the-version-graph-has-no-solution) |
| Segfault / C-level crash when two frameworks integrate | [4 — the version graph has no solution](#pattern-4-the-version-graph-has-no-solution) |
| Crash *during CUDA graph capture*, fine in eager mode | [5 — capture demands static everything](#pattern-5-capture-demands-static-everything) |
| "cross-device tensor" error at TP > 1 | [5 — capture demands static everything](#pattern-5-capture-demands-static-everything) |
| HTTP 400 on long inputs | [6 — a boot-time constant, a runtime bug](#pattern-6-a-boot-time-constant-a-runtime-bug) |
| Your judge reports quality loss you cannot reproduce by eye | [6 — a boot-time constant, a runtime bug](#pattern-6-a-boot-time-constant-a-runtime-bug) |
| `loss = 0`, `grad_norm = nan`, from step one | [7 — uninitialised memory meets low precision](#pattern-7-uninitialised-memory-meets-low-precision) |

---

## Engine index

Same incidents, grouped by where they bit — this is how you'd tell the story in
an interview ("the vLLM stuff," "the SGLang stuff," "the training thing").

### vLLM
- **I1** `VLLM_ATTENTION_BACKEND` env var accepted, ignored, silent
- **I4** `import vllm` works in shell, fails in cluster job
- **I6** FlashInfer JIT compiles at runtime, needs dev headers the runtime image lacks
- **I7** `flash_attn` wheel doesn't match B200 arch — plain `ImportError`
- **I8** Ray + vLLM + transformers version graph has no solution
- **I9** TRL's GRPOTrainer segfaults inside vLLM process
- **I13** `--max-model-len 32768` rejects docs at 33K tokens

### SGLang
- **I2** `enable_thinking=False` does nothing; template strips prefix
- **I3** `SGLANG_EXTERNAL_MODEL_DIR` never existed — package var, registry subprocess gap
- **I5** Model runner subprocess doesn't inherit `PYTHONPATH`
- **I12** TP=2 cross-device error during CUDA graph capture
- **I14** (shared) Harness truncation masquerades as quality regression

### Training (surfaced via serving)
- **I15** Fresh classification head, bf16, uninitialised weights → silent `NaN`

### Platform / harness
- **I10** Ray submit hangs forever — namespace permission, not code
- **I11** Custom model boolean indexing breaks graph capture
- **I14** Judge reports `information_loss` on truncated outputs

---

## Pattern 1: The knob that does nothing

> **Scene.** Monday 3pm. You're debugging a JIT failure from FlashInfer (I6). You
> set `VLLM_ATTENTION_BACKEND=FLASHINFER` to force the backend. Server starts
> clean. Logs show it's still using the default backend. You try `FLASH_ATTN`,
> `XFORMERS` — same. The variable is *accepted*. It just does nothing. Three
> hours later you're reading vLLM source and find the V1 engine doesn't read
> that variable at all.

The worst failures in this list are not crashes. They are settings that are
**accepted, ignored, and never mentioned again.** Nothing errors. The system
starts. It simply does something other than what you asked, and you spend an
afternoon debugging the consequence instead of the cause.

### I1: `VLLM_ATTENTION_BACKEND` is ignored by the V1 engine

**What you see.** You want a specific attention backend — say, to dodge a JIT
failure. You set `VLLM_ATTENTION_BACKEND`. The server comes up. It uses a
different backend anyway.

**What it was.** vLLM's V1 engine does not honour that environment variable.
It is a V0-era control, still present, still settable, no longer wired to
anything on the path that matters.

**Why it happens — the V1 rearchitecture.** vLLM V1 replaced the old
`VLLM_ATTENTION_BACKEND` env var with a structured **AttentionConfig** system.
The old variable was deprecated in #26315 (Dec 2025) and fully removed in
#32812 (Jan 2026). The literal string `VLLM_ATTENTION_BACKEND` no longer
appears anywhere in the vLLM 0.26+ source tree — nothing reads it.

**How V1 automatic backend selection works.** When you don't specify a
backend, V1 iterates through registered backends in **priority order** and
picks the first one that validates against your configuration (model dtype,
head size, compute capability, KV cache dtype, block size, attention type):

```
Priority (CUDA, standard attention):
  1. FLASHINFER      — SM90+ decode, FlashInfer-native path
  2. XQA             — SM90 TRT-LLM decode path (via FlashInfer)
  3. TRTLLM_GEN      — SM100+ (Blackwell) decode, supports sinks
  4. FLASH_ATTN_4    — SM100+ default prefill
  5. FLASH_ATTN_3    — SM90 (Hopper) default prefill
  6. FLASH_ATTN_2    — fallback prefill
  7. TRITON_ATTN     — Triton backend, broad compatibility
  8. TRITON_MLA      — MLA decode
  9. FLASHMLA        — DeepSeek-style MLA decode
 10. ... (ROCm/CPU have separate lists)
```

Each backend implements `validate_configuration()` checking: compute capability,
supported dtypes (fp16/bf16/fp32), KV dtypes, block sizes, head sizes, sink
support, non-causal, sparse, multimodal prefix, DCP, attention types. The
first compatible backend wins. If none validate, you get an error listing *all*
backends and their rejection reasons.

**The current mechanisms (what actually works):**

| Method | Example |
|--------|---------|
| CLI flag | `vllm serve model --attention-backend FLASH_ATTN` |
| Structured config | `vllm serve model -ac.backend FLASH_ATTN` or `-ac '{"backend": "FLASH_ATTN"}'` |
| Python `attention_backend` kwarg | `LLM(model="...", attention_backend="FLASH_ATTN")` |
| Python `AttentionConfig` | `LLM(model="...", attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN))` |

All four paths converge to the same validation logic. Explicit selection bypasses
auto-selection and validates *only* your choice — if incompatible, you get a
specific error (`Selected backend FLASHMLA is not valid: compute capability not
supported`).

**GitHub issues tracking this:**

- **#50292** — "VLLM_ATTENTION_BACKEND env var is silently ignored —
  `attention_backend=` kwarg is the only mechanism that works" (open). The
  reporter caught it only because a controlled experiment showed zero effect
  across paired comparisons.
- **#50346** (open PR) — adds an actionable warning when the removed var is
  set: `Environment variable VLLM_ATTENTION_BACKEND is no longer read by vLLM
  and has no effect; use the --attention-backend CLI flag or the
  attention_backend argument to LLM()/EngineArgs instead.` (merged after
  0.26.0; backported to 0.26.1+).

**The fix.** Stop trying to steer the backend via env var. Either:
1. **Fix the root cause** of why the default backend fails (e.g., FlashInfer
   JIT missing headers → install dev headers, Pattern 3), or
2. **Use the supported mechanism** — `attention_backend="FLASH_ATTN"` in
   `LLM()` / `EngineArgs`, or `--attention-backend` on CLI.

**How to spot it next time.** When a documented knob appears to do nothing,
**grep the installed source for the variable name** before you believe the
docs. `grep -rn VLLM_ATTENTION_BACKEND .venv/lib/python3*/site-packages/vllm/`
answers in seconds what an afternoon of experiments will not: whether anything
reads it at all, and on which code path.

> **Tip.** The V1 docs at `docs/design/attention_backends.md` (auto-generated
> from the backend registry) list the exact priority tables and validation
> rules for your hardware. Read that before forcing a backend.

### I2: `enable_thinking` is cosmetic on Qwen3

**What you see.** Qwen3 emits garbled or missing `think` tags. You pass
`enable_thinking=False`. Nothing changes.

**What it was.** Three separate problems wearing one costume.

1. Qwen3's chat template **unconditionally injects** `think`. The kwarg does
   not gate it; the template does not consult it.
2. SGLang strips the template prefix from the response, so the *opening*
   `think` never reaches you — leaving output that looks like a malformed
   thinking block.
3. When `enable_thinking=False` *is* respected, the template emits an **empty
   thinking block** (`think\n\nthinking\n\n`) — and SGLang's `Qwen3Detector`
   reasoning parser strips the opening `think` but leaks the closing
   `thinking` into `content`. Output: `"thinking\n\nThe answer is 42."`

**Why it happens.** A chat template is a Jinja program shipped inside the
tokenizer, not part of the serving engine. A kwarg only does something if the
template author wrote a branch for it. "The API accepts this argument" and
"the model's template respects this argument" are unrelated facts. The
stripping is done by SGLang's reasoning parser (`reasoning_parser.py`,
`Qwen3Detector`), which expects `think`…reasoning…`thinking` — a format the
empty-block case doesn't match.

**Peek at the system.** The request path is:
`ChatCompletionRequest.normalize_reasoning_inputs()` (maps top-level
`enable_thinking` → `chat_template_kwargs`) → `TemplateManager.apply_chat_template()`
→ Jinja renders → model generates → `Qwen3Detector` strips tags.

- **SGLang PR #33155** — "honor top-level enable_thinking field": before it,
  bare `{"enable_thinking": false}` in the request body was **silently dropped
  by Pydantic** (the field wasn't declared), so the kwarg never even reached
  the template.
- **SGLang #6675** — `enable_thinking=False` breaks structured JSON outputs.
- **QwenLM/Qwen3.6#90** — vLLM has `--default-chat-template-kwargs` for
  server-level defaults; SGLang lacks the equivalent.

**The fix.** Prefix `think\n` explicitly in the scripts that need it. For the
judge model — where thinking is unwanted — pass
`chat_template_kwargs: {"enable_thinking": false}` *and* state the required
output format explicitly in the system prompt, so correctness does not depend
on the template cooperating.

**How to spot it next time.** Render the template yourself and read the actual
string the model receives:

```python
print(tokenizer.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True))
```

Do this before debugging model behaviour. It takes ten seconds and it is
ground truth.

> Lecture 24 says a mis-applied chat template *"produces subtly worse output
> that looks like a model problem."* This is that sentence, in the wild —
> see [24. Serving](24-serving.md#the-whole-path-and-where-time-goes).

### I3: `SGLANG_EXTERNAL_MODEL_DIR` was deprecated silently

**What you see.** You follow a guide, set `SGLANG_EXTERNAL_MODEL_DIR`, and your
custom model is not found.

**What it was.** The name is wrong in a more interesting way than "deprecated":
that variable **never existed**. SGLang only ever had
`SGLANG_EXTERNAL_MODEL_PACKAGE` (since PR #13429, ~March 2025), which names a
Python package, not a directory. The blog post you were following had drifted
from the API.

**Why it happens.** Fast-moving projects rename their extension points. Search
results and blog posts outlive the API they describe, and an unread env var
fails silently by construction. Worse: the *correct* variable,
`SGLANG_EXTERNAL_MODEL_PACKAGE`, is read by the **main process** — but the
model actually runs in a **subprocess**, and there the package must be
importable via `sys.path`. The registry itself is the tip of a much deeper
import chain (the full story is [I5](#i5-an-sglang-subprocess-cannot-import-your-custom-model)).

**Peek at the system.** The extension mechanism:

- `python/sglang/srt/model_loader/` — the loader builds model classes by
  looking up a **registry** (`ModelRegistry`) by `__name__`, matching the
  architecture string in the HF `config.json` (e.g. `Qwen2ForCausalLM`).
- `ModelRegistry.register("sglang.srt.models")` registers the built-ins; when
  `envs.SGLANG_EXTERNAL_MODEL_PACKAGE` is set, it calls
  `ModelRegistry.register(external_pkg, overwrite=True)` instead, then imports
  the package and collects every module that has an `EntryClass` attribute.
- **Docs PR #21050** added the *documentation* for the env var long after the
  code existed — the naming confusion goes all the way up.

**The fix.** Use the package variable — though on this project even that did
not survive contact (see [pattern 2](#pattern-2-which-python-is-installed)).

**How to spot it next time.** Check the installed version's own source, not the
internet. `grep -rn 'EXTERNAL_MODEL' .venv/.../sglang/` tells you which names
the code you are actually running still reads.

!!! note "The general lesson"
    All three failures share a shape: **configuration that is accepted but not
    read.** Type-checked arguments and validated config files protect you from
    *malformed* settings, not from *ignored* ones.

    The habit worth building: when a knob appears not to work, stop testing
    outcomes and go read the code path. The question "does anything read this?"
    is cheap to answer directly and expensive to answer by experiment.

---

## Pattern 2: Which Python is "installed"?

> **Scene.** You installed vLLM. It imports in your terminal. You submit a job
> to the cluster and it dies on `import vllm` — same machine, same disk, same
> `pip install`. The error doesn't tell you *which* Python. That's the bug.

"I installed it" is not a fact about your machine. It is a fact about **one
interpreter**. When another process runs a different interpreter, the package
is not there — and the error message is the same `ModuleNotFoundError` you
would get if you had never installed anything.

### I4: `import vllm` fails inside cluster job scripts

**What you see.** vLLM is installed. It imports fine in your shell. A job
submitted to the cluster dies on `import vllm`.

**What it was.** `uv add` installed into the project's `.venv`. The cluster's
job runner launched scripts with the system interpreter — a base conda
install — which has its own `site-packages` and has never heard of vLLM.

**Why it happens.** There is no global "installed" state on a machine. There is
a set of interpreters, each with a `site-packages`. Activating a virtualenv is
just prepending a directory to `PATH` for *your* shell; a job runner that
invokes an absolute interpreter path never sees it.

**Peek at the system.** This is compounded by the engine's own launcher
habits. vLLM's `env_setup.py` and SGLang's `engine.py` both call
`multiprocessing.set_start_method("spawn")` — the child process starts from a
**fresh interpreter**, `sys.path` rebuilt, nothing inherited except the
environment. `VIRTUAL_ENV` and `PATH` pointing at your venv are not enough;
the child looks up `vllm` on its *own* `sys.path`, which starts with the
parent's `sys.executable`'s site-packages. Two interpreters is bad enough;
two interpreters *inside the engine* is three ways to be wrong.

- **vLLM #15461** — the long-running saga of "collect_env doesn't detect
  virtual environments", which is the same confusion fossilised: the tooling
  that reports your environment also assumes one interpreter.

**The fix.** Make the entrypoint select the interpreter explicitly:
`uv run python main.py`, so the venv is active inside the job rather than
merely in the shell that submitted it.

```
   your shell                    the job runner
   ──────────                    ──────────────
   PATH → .venv/bin/python       hardcoded /opt/conda/bin/python
          └─ site-packages/         └─ site-packages/
             └─ vllm ✓                 └─ (no vllm) ✗

   same machine, same disk, same "pip install" — different answer
```

**The fix.** Make the entrypoint select the interpreter explicitly:
`uv run python main.py`, so the venv is active inside the job rather than
merely in the shell that submitted it.

**How to spot it next time.** Print the interpreter from *inside* the failing
process, not from your terminal:

```python
import sys; print(sys.executable, sys.path)
```

This single line resolves nearly every "but it's installed" bug immediately.

### I5: An SGLang subprocess cannot import your custom model

**What you see.** A custom model package that imports perfectly in your own
process is invisible to SGLang's model runner.

**What it was.** SGLang 0.5.13 spawns the model runner as a **subprocess** that
does not inherit the parent's `PYTHONPATH`. Symlinks, `.pth` files and
environment variables were all tried; none propagated.

**Why it happens.** Serving engines run the model in a separate process on
purpose — Lecture 24's argument for keeping the API server's event loop away
from the engine loop. That isolation is a feature, and it isolates your import
path along with everything else. How a child process is spawned (`fork` vs
`spawn`, whether the environment is copied, whether `sys.path` is rebuilt) is
an implementation detail of the parent that you do not control.

**Peek at the system.** The chain, in order:

1. `engine.py` calls `mp.set_start_method("spawn", force=True)`. On Linux the
   default would be `fork`, which copies the parent's memory *and* its
   `sys.path` — and after CUDA initialises, fork is unsafe (the CUDA context
   cannot be inherited by a fork child reliably). SGLang forces spawn for
   safety.
2. A **spawn** child is a fresh interpreter: it imports your entrypoint module
   again, re-executes your `sys.path` manipulation, but **does not inherit the
   parent's runtime `sys.path` or `PYTHONPATH`** — nothing from the parent's
   process memory survives. Environment variables *do* pass through, but an
   env var naming a package does not make the package importable.
3. The runner then imports your model package through the *registry*
   ([I3](#i3-sglang_external_model_dir-was-deprecated-silently)) — if the
   package can't be found on the child's path, the error surfaces as
   "architecture not registered", which looks nothing like an import error.

So the debugging difficulty is structural: the error is reported far from the
cause, and the cause (spawn semantics) is invisible in the error text.

**The fix.** Stop fighting the boundary and put the code where the child will
look anyway: patch SGLang's installed model source in place, so the runner
loads it through its own normal import path.

That is an unpleasant fix and worth naming as such — it is edit-in-place on a
dependency, and it must be re-applied after every upgrade. It was chosen
because the supported extension point did not work
([pattern 1](#pattern-1-the-knob-that-does-nothing)) and shipping is a
constraint. If you do this, keep the patch as a **script**, not a manual edit,
so it is reproducible and reviewable.

**How to spot it next time.** When something works in your process and not in
the engine's, ask *which process is failing* before asking what is failing.
Print `sys.executable`, `sys.path` and `os.environ` from inside the child.

> [24. Serving](24-serving.md#decouple-the-api-server-from-the-engine-loop)
> explains why engines run the model in a separate process at all — the
> isolation that helps you here is the same isolation that broke this import.

---

## Pattern 3: The container is not the machine

> **Scene.** vLLM runs fine. You change TP size and it crashes with
> `cublasLt.h not found`. You didn't change code. You changed a *config*. The
> error mentions a header file — at runtime. The container was fine until it
> wasn't.

Some Python packages are not really installed when you install them. They
carry source that gets compiled **the first time you run it**, against
**your** GPU and **your** CUDA toolkit. A runtime image that lacks a compiler
or headers is fine right up to the moment something decides to build.

### I6: FlashInfer's JIT cannot find CUDA headers

**What you see.** vLLM 0.22.1 runs, then crashes when you change TP size.
FlashInfer's JIT reports it cannot find `cublasLt.h`, `nvrtc.h`, `curand.h`.

**What it was.** The CUDA 12.8 *runtime* image has no CUDA **dev** headers.
FlashInfer JIT-compiles attention kernels per shape, and shapes change with TP
degree — so the compile is triggered by a config change rather than at startup.

**Why it happens.** This is a build-time dependency that has been deferred to
runtime. Two things follow, and both are counter-intuitive:

- **It is fine until it isn't.** A cached kernel means no compile means no
  error. Changing TP invalidates the cache and the failure appears from a
  change that looks unrelated to compilation.
- **A "runtime" image is exactly the wrong image** for a library that compiles
  at runtime. `-runtime` and `-devel` container tags encode an assumption —
  that compilation already happened — which JIT breaks.

**Peek at the system.** The JIT is *per-shape*, and the shape key includes
`num_kv_heads` — which changes with the TP degree. That is the concrete link
between "I changed a config" and "it tried to compile":

```
   FlashInfer kernel cache key (per shard):
   (batch_size, num_kv_heads, head_dim, page_size, dtype, KV_layout, ...)

   TP=2:  num_kv_heads per GPU = 4
   TP=4:  num_kv_heads per GPU = 2     ← different key → MISS → JIT
```

The cache itself lives under `~/.cache/flashinfer/` (subdirectories per
compiler/config), so a fresh container also means a cold cache — the compile
happens on the very first request that needs a kernel, not at startup.

How vLLM decides: `has_flashinfer_cubin()` in `vllm/utils/flashinfer.py`
probes for prebuilt cubins; if absent, vLLM itself kicks off FlashInfer's
`flashinfer.jit` — and that's when the missing headers bite.
**vLLM #42291** is the canonical report of this exact failure mode.

**The fix.** Two options, both legitimate:

1. **Lock the shapes** so the cached kernels stay valid (TP=4 for one model,
   TP=2 for another) and invoke the entrypoint directly rather than through a
   wrapper that might re-resolve the environment.
2. **Install the headers**: `ninja-build`, `cuda-nvrtc-dev-12-8`,
   `libcublas-dev-12-8` — making the runtime image capable of the compile it
   was always going to attempt. (vLLM's own Dockerfile does exactly this: the
   *runtime* image installs the *devel* packages for FlashInfer and
   flash-attn, precisely because they JIT at runtime.)

**How to spot it next time.** If a stack trace mentions `.h` files, `nvcc`,
`ninja` or a build directory, you are looking at a **compile** failure wearing
a runtime failure's clothing. Ask what triggered a build now, and what changed
to invalidate the cache.

### I7: `flash_attn.ops` fails to import on a new GPU generation

**What you see.** vLLM 0.23.0 is installed and refuses to serve;
`import flash_attn.ops` fails.

**What it was.** The PyPI `flash-attn` wheel was not built for B200's
architecture and CUDA version.

**Why it happens.** A wheel is compiled binary. It carries kernels for the
architectures it was built for and an ABI it expects. New silicon arrives
before prebuilt wheels do, and the failure surfaces as a plain `ImportError` —
which reads like a missing package rather than an incompatible binary.

**Peek at the system.** The GPU generations matter more than the release
number. `flash-attn` 2.8.3 (Aug 2025) ships kernels for SM70–SM110 (V100
through Blackwell *datacenter*), which includes B200 = SM100. But consumer
Blackwell (RTX 5090 = SM120) is a *different architecture* — and SM120 kernel
support was only merged upstream in March 2026 (PRs #2329, #2330, #2333),
with **no wheel released since**. A `pip install flash-attn` on an SM120
machine therefore installs a wheel with no kernels for the card in front of
it, and every `import flash_attn.ops` fails at link time.

```
   B200 / GB200        SM100   Blackwell datacenter    covered by 2.8.3  ✓
   RTX 5090 / 5080     SM120   Blackwell consumer      not in any wheel ✗
```

The error message is the giveaway: `ImportError ... no kernel image is
available for execution on the device` — "kernel image", not "package".

**The fix.** Use a prebuilt wheel matching the architecture and ABI, or switch
that cluster to SGLang, which did not need the extension.

**How to spot it next time.** On new hardware, treat every compiled dependency
(`flash-attn`, `flashinfer`, custom CUDA ops) as suspect first. Check
`torch.cuda.get_device_capability()` against what the wheel advertises before
debugging anything upstream.

!!! note "The general lesson"
    Pattern 2 and pattern 3 are both **environment** failures, and both were
    initially mistaken for library bugs.

    A useful reflex: when something fails only in one place, enumerate what
    differs about that place — interpreter, process, image, GPU architecture —
    before forming any theory about the library itself.

---

## Pattern 4: The version graph has no solution

> **Scene.** You need Gemma 4. It needs `transformers==5.10.2`. Ray 2.55 needs
> `vllm>=0.18` which needs `transformers<5`. The resolver sits there. You
> upgrade, downgrade, pin, unpin — nothing resolves. The constraint graph is
> *provably empty*. Another day of resolver attempts won't help.

Serving stacks pin each other transitively. Once four projects each constrain
the others, "upgrade until it works" can be **provably impossible** — and
recognising impossibility early is worth more than another day of resolver
attempts.

### I8: Ray's LLM serving layer versus the model you need

**What you see.** The managed `ray.serve.llm` / `LLMConfig` path will not
resolve.

**What it was.** A genuine deadlock:

```
   Ray 2.55.x        requires  vllm >= 0.18
   vllm >= 0.18      requires  transformers < 5
   the target model  requires  transformers == 5.10.2
                                ────────────────────────
                                transformers < 5  AND  == 5.10.2
                                no solution exists
```

Ray also vendors its own vLLM fork, so imports do not align with upstream even
when versions nominally do.

**Why it happens.** Every layer pins the layer below to protect its users from
breakage. Stack four such layers and the *intersection* of their constraints
can be empty. No amount of resolver cleverness helps; the requirement is
contradictory.

**Peek at the system.** The specific chain in this incident was:

- **Ray 2.55.x** requires `vllm>=0.18` — but Ray ships its *own fork* of vLLM
  inside `ray.serve.llm`, so "vLLM" means two different codebases depending on
  import path.
- **vllm>=0.18** requires `transformers<5`.
- The **target model** (Gemma 4) needs `transformers==5.10.2` — its
  `model_type` (`gemma4`) only exists from transformers 5.5.

So the empty intersection is not an accident of one version; it is a
**release-cadence skew**: Gemma 4 shipped before Ray's managed layer was
updated, and no patch-level bump on either side can fix it.

- **ray-project/ray#60780** and **#62497** — managed-LLM integration reports
  around exactly this friction.
- **vllm-project/vllm#39216** — transformers 5.x compatibility tracking.

**The fix — and this is the transferable part.** Stop trying to satisfy the
constraint and **remove the coupling**. The managed integration was abandoned
in favour of composing the pieces directly:

```
   BEFORE  ray.serve.llm  ──manages──►  vLLM
           (one dependency graph, no solution)

   AFTER   cluster submit ──► manual `vllm serve` on the head node,
                              split by CUDA_VISIBLE_DEVICES
                          ──► job submit dispatches work over HTTP
           (two processes, one HTTP boundary, two independent graphs)
```

An HTTP boundary between components is also a **dependency boundary**. Each
side resolves its own versions and neither constrains the other.

**How to spot it next time.** When a resolver fails, write the constraint chain
out by hand. If it is contradictory rather than merely awkward, the answer is
architectural — decouple — not another version bump.

### I9: An RL trainer's vLLM integration segfaults

**What you see.** `trl vllm-serve` segfaults; TRL 1.6 warns about vLLM 0.23
compatibility.

**What it was.** The `GRPOTrainer` ↔ vLLM integration was unstable at those
versions — a C-level crash, not a Python traceback.

**Why it happens.** Two independently-versioned projects sharing a process,
CUDA context and memory pool. When the contract between them shifts, you get a
segfault rather than an error, because the failure is below Python.

**Peek at the system.** How the two actually share state in "colocate" mode
(`trl vllm-serve` with `WorkerExtension`):

```
   TRL process ──spawns──► vLLM engine workers (multiprocessing)
                              │
                              ├─ same NCCL communicator
                              ├─ same CUDA context & memory pool
                              │   (vLLM's MemoryPool: torch blocks owned by
                              │    the engine, borrowed by TRL, freed on exit)
                              └─ pynccl_comm updates model weights in place
                                 (update_named_param: in-place copy into
                                  the engine's KV cache and model buffers)
```

On shutdown, both sides free memory they believe is theirs — the classic
double-free, invisible in Python, fatal at the CUDA level. Two documented
variants of the same seam:

- **huggingface/trl#3671** — GRPO + vLLM colocate + PEFT hangs with
  `is_cpu` errors; the known workaround is
  `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`, i.e. change the transport and the
  bug moves.
- **vllm-project/vllm#16993** — "SegFault on exit" from vLLM's own side of
  the shared memory pool.

**The fix.** Drop the built-in integration and compose explicitly: SGLang
serving rollouts over HTTP, a separate large judge model scoring them, and a
manual training step. More moving parts, each independently debuggable and
independently versioned.

**How to spot it next time.** A **segfault at an integration seam** is rarely
your bug to fix. Check whether the two projects claim to support each other's
versions before debugging further — and prefer a process boundary over a
shared address space when both sides move fast.

### I10: When the platform is the problem

One incident in this project was not technical at all: a managed serving
submission hung in `PROVISIONING` indefinitely because the required namespace
was not enabled for the project — an org-admin permission.

**Peek at the system.** `PROVISIONING` is a **control-plane** state: the
platform's scheduler is claiming a GPU node for you. Code never runs in this
state — there is no log, no traceback, no process. The platform control plane
and the data plane are separate systems; a hang here means the *first* of
them couldn't finish a bookkeeping step (allocating the namespace's GPU
quota), which no amount of application-level debugging can reach.

**The fix.** Confirm entitlement first — with the platform owner, not with
logs. The question is "is this *supposed* to be able to run here?", and that
is answered by a permission check, not a profile.

**How to spot it next time.** It is here for one reason: **an infinite hang
with no error is a signal in itself.** Crashes come from code that ran; silent
hangs often come from code that never started. Check that you are entitled to
the resource before profiling the thing that is not running. A useful litmus:
if *every* submission to that namespace hangs at the same stage, the problem
is not your code.

---

## Pattern 5: Capture demands static everything

> **Scene.** Your custom model works in eager mode. You enable CUDA graphs and
> it crashes *during capture*. The line? `residual[:, mask] = 0.0`. It's just
> a masked assignment. But capture records commands, not decisions — and which
> addresses get written depends on the *contents* of `mask`. That means a
> device-to-host sync inside the captured region. Not allowed.

[Lecture 13](13-cuda-graphs.md#the-constraint-that-shapes-everything) states the
rule: a captured graph replays **exactly the same operations on exactly the same
memory addresses**, and capture therefore fixes three things — the shapes, the
addresses, and the sequence of commands.

Both incidents here are that constraint being violated. The first violates a
fourth thing the lecture does not list.

### I11: Indexed assignment breaks graph capture

**What you see.** A custom model works in eager mode and crashes during CUDA
graph capture. The offending line:

```python
residual[:, mask] = 0.0        # mask is a boolean tensor
```

**What it was.** Indexed assignment with a data-dependent mask cannot be
captured.

**Why it happens — and this is the interesting part.** Capture records
*commands*, not decisions. `residual[:, mask] = 0.0` cannot be turned into a
fixed command list, because **which addresses get written depends on the
contents of `mask`** — and reading those contents means a device-to-host sync,
inside a region where nothing may touch the host.

```
   residual[:, mask] = 0.0
   ├─ read mask's VALUES        → device-to-host sync  ✗ not capturable
   ├─ compute which columns     → depends on runtime data
   └─ write only those          → address set varies per call

   residual = residual * binary_mask
   ├─ read both tensors         → fixed addresses      ✓
   ├─ multiply elementwise      → fixed command        ✓
   └─ write every element       → fixed address set    ✓
```

So Lecture 13's three constraints have a fourth sibling: **no host
synchronisation.** Shapes, addresses, command sequence — and nothing that
requires the CPU to look at a value mid-graph.

The full un-capturable list is a checklist worth keeping:

```
   data-dependent control flow
   ├─ boolean indexing / indexed assignment   (this incident)
   ├─ .item() or .cpu()                        → implicit device-to-host copy
   ├─ if tensor > 0:  in Python                → needs a value on the host
   ├─ torch.nonzero(...) / torch.argwhere      → output size is data-dependent
   └─ early exit on an EOS check               → changes the command count
```

Any of these forces a synchronisation that capture cannot record — the graph
is a frozen script of launches, and a launch whose arguments depend on a value
the GPU just computed is not expressible in it.

**The fix.** Replace the conditional write with an unconditional arithmetic
one. Pre-allocate a tensor of ones with zeros at the masked indices, then:

```python
residual = residual * binary_mask
```

Same result. Every address is written every time, so the command list is fixed.
It does strictly more arithmetic — and is faster, because it is capturable.
That is the same trade FlashAttention makes: **more math to avoid a memory or
synchronisation cost.**

**How to spot it next time.** Anything whose *control flow* depends on tensor
**contents** is un-capturable: boolean indexing, `.item()`, `if tensor > 0`,
`torch.nonzero`, early exit on an EOS check. Rewrite as arithmetic over a fixed
address set — masks become multiplies, branches become `torch.where`.

### I12: TP=2 crashes during capture

**What you see.** SGLang with `tensor_parallel_size=2` fails with a
cross-device tensor error on B200.

**What it was.** TP > 1 introduces cross-device operations that failed during
graph capture on that platform.

**Why it happens.** TP inserts a collective into every layer — Lecture 22's
all-reduce. Capturing a graph that spans devices means recording operations
whose completion depends on *another GPU*, which multiplies the ways capture
can go wrong.

**Peek at the system.** Count the collectives. A 28-layer transformer with
TP=2 performs **two all-reduces per layer** — one after the attention output
projection, one after the MLP down-projection — for **56 collectives per
forward pass**. Each is an NCCL launch that synchronises two GPUs and, during
capture, must itself be captured as a unit — and CUDA graphs do not play well
with NCCL communicator state (the collective is a *cooperative* launch across
devices whose ordering is managed by the peer, not by the capturing stream).
On B200 the failure surfaced as a cross-device tensor error during capture.

**The fix, and why it was the right call.** TP=1, and scale with **data
parallelism** instead — multiple independent single-GPU instances.

This is worth understanding rather than memorising, because the reasoning is
specific and it generalises. From
[Lecture 22](22-tensor-parallelism.md#why-scaling-isnt-linear): TP is a
**latency** optimisation. It splits one token's work across GPUs so a single
sequence finishes sooner. It costs a collective per layer, and on this workload
that cost is dominated by the fixed per-collective overhead rather than
bandwidth.

With 183 GB of VRAM per B200, the model fits on one card — so TP's *other*
justification, "the model doesn't fit", does not apply either. And for offline
batch work, **throughput is the objective, not per-token latency**:

```
   TP=2    one model across 2 GPUs      halves per-token latency (in theory)
                                         pays a collective every layer
                                         needs cross-device capture to work

   DP      two independent instances    doubles throughput
                                         zero collectives
                                         each GPU captures its own graph
```

For throughput-bound work, data parallelism is not the fallback — it is the
better answer, and it happens to sidestep the crash entirely.

**How to spot it next time.** Ask what you are actually optimising before
reaching for TP. If the answer is throughput and the model fits on one device,
**replicas beat sharding**: more work per second, no collectives, no
cross-device capture.

---

## Pattern 6: A boot-time constant, a runtime bug

> **Scene.** You set `--max-model-len 32768` at startup. Weeks later, a
> 33,000-token document hits the server and gets a 400. The limit wasn't a
> suggestion — it sized the KV cache. Or: your judge reports `information_loss`
> on outputs that look fine. The harness defaulted to `max_new_tokens=2048`.
> Long answers were cut mid-sentence. The model was fine; the *measurement*
> was broken.

Both of these are a number chosen once, at startup, that becomes a
**correctness** problem much later — and in the second case, a number that
silently corrupted an evaluation.

### I13: `--max-model-len` too small: HTTP 400 on long documents

**What you see.** The server returns 400 on long documents.

**What it was.** Started with `--max-model-len 32768`; the documents exceeded
32K tokens.

**Why it happens.** `max_model_len` is not merely a validation limit — it sizes
the KV cache. From [Lecture 05](05-kv-cache.md#what-it-costs), an engine
reserves cache capacity per sequence, and paged engines carve VRAM into a block
pool up front. Raising the limit means fewer sequences fit concurrently, so the
value is a **real throughput trade**, not a formality.

**Peek at the system.** Sizing is a concrete arithmetic problem:

```
   KV bytes per token = 2 (K+V) × n_layers × n_kv_heads × head_dim × bytes

   Qwen3-0.6B (28 layers, 2 KV heads, d=128) in fp16:
   2 × 28 × 2 × 128 × 2 B  =  28,672 B/token  ≈  28 KB/token

   32K tokens  →  ~917 MB of KV cache for ONE sequence
   64K tokens  →  ~1.8 GB
   128K tokens →  ~3.7 GB
```

SGLang's `KVCache` allocates `(max_seqs, max_seq_len, n_kv_heads, head_dim)`
per layer up front; vLLM's paged pool grows the same way. The paged
architecture (Lecture 09) saves only the *unused tail* (at most
`block_size − 1` tokens per sequence) — the per-token cost is identical, so a
generous `max_model_len` is a real VRAM commitment even if no long request
ever arrives. That is the honest tension: too low and long requests are
rejected; too high and concurrency drops for every request. When memory gets
tight, engines preempt
([Lecture 09](09-paged-attention.md#preemption)); a limit set far above your
real p99 spends VRAM that could have been batch capacity.

**The fix.** Restart with `--max-model-len 65536`, and pick the number from
the p99 of your input length distribution — not the mean.

### I14: `max_new_tokens` truncation manufactured a quality failure

**What you see.** A judge model reports `information_loss` on outputs that look
fine when you read them.

**What it was.** The evaluation harness defaulted to `max_new_tokens=2048`. Long
outputs were cut off mid-sentence. The judge — correctly — flagged truncated
text as incomplete.

**Why it happens.** The model was fine. The **harness** was broken, and it
failed in the most expensive possible direction: it produced plausible,
well-formed, *wrong* results rather than an error. A truncation limit is
invisible in the output; a cut-off answer looks exactly like a model that
stopped early.

**Peek at the system.** Engines expose the truth if you look: every output
carries a `finish_reason`. `"length"` means "hit the token ceiling",
`"stop"` means "ended on a stop token". On OpenAI-compatible endpoints the
same field is `finish_reason` in the response `choices`. The whole class of
bug collapses into one check:

```python
truncated = sum(1 for o in outputs if o.finish_reason == "length")
print(f"{truncated}/{len(outputs)} hit the token limit")
```

**The fix.** A hard rule of `max_new_tokens=16384` for evaluation runs.

**How to spot it next time.** Before believing any quality regression, run the
`finish_reason` check above. If that number is not zero, fix the harness
before analysing the model.

> This is the single most useful entry on this page for interviews, because it
> is about **trusting your measurement apparatus**. The
> [field notes](field-notes.md) record the same lesson from another operator:
> holding the model fixed and changing *only* the harness swung a long-horizon
> score by up to 18 points. [Lecture 19](19-quantization.md#measuring-what-it-costs)
> makes the matching point about quality metrics — perplexity is necessary and
> not sufficient.

---

## Pattern 7: Uninitialised memory meets low precision

> **Scene.** Training a 0.6B model with a sequence-classification head. Loss
> is exactly `0`, gradient norm is `nan`, from step one. You assume bf16 is
> fragile. It's the opposite: bf16's *wide range* is what let the garbage
> survive long enough to do damage.

### I15: `loss = 0`, `grad_norm = nan` on a fresh classification head

**What you see.** Training a 0.6B model with a sequence-classification head:
loss is exactly `0`, gradient norm is `nan`, from the first step.

**What it was.** `AutoModelForSequenceClassification` adds a fresh `score`
head. Its weights were not properly initialised, and in bf16 the values it
picked up were garbage on the order of `7.2e11`.

**Why it happens — and the obvious explanation is wrong.** The tempting answer
is "bf16 is fragile." The truth is nearly the opposite, and it is a better
interview answer.

bf16 keeps **fp32's exponent range** (8 exponent bits, max ≈ `3.4e38`) and
sacrifices mantissa bits instead. So:

```
   value 7.2e11

   in fp16   max finite is 65504     →  overflows to inf IMMEDIATELY
                                          you find out at once

   in bf16   max finite is ~3.4e38   →  perfectly representable
                                          propagates as a FINITE number
```

The garbage survives the cast, and that is what makes it dangerous: it
propagates as an ordinary finite number instead of announcing itself.

Where it finally breaks is the **matmul accumulation**, not the softmax. A
logit is a sum of 1024 products. With weights of that magnitude the running sum
climbs until it exceeds even bf16's range, becomes `inf`, and the first
`inf - inf` in the backward pass produces `NaN` — which then poisons every
gradient it touches, giving `nan` grad norm while the loss degenerates.

**So bf16's wide range is precisely what made this silent.** fp16 would have
overflowed on the very first cast and failed louder and sooner. bf16's range is
exactly why it is preferred for training — it resists the overflow that plagues
fp16 — and that same property let uninitialised garbage travel several
operations before detonating somewhere that looks unrelated to its cause.

!!! note "Verified, with one correction"
    Checked in PyTorch while writing this. `7.2e11` casts to
    `721554505728.0` in bf16 and to `inf` in fp16 — the inversion above is real
    (`torch.finfo` gives max finite `3.39e38` for bf16, `65504` for fp16).

    One thing worth stating precisely, because it is easy to get wrong: garbage
    of *uniform* magnitude does **not** NaN. `softmax` subtracts the row max, so
    even a `2.4e13` logit yields a clean `[1.0, 0.0]`. It takes garbage that
    varies enough to push the accumulation past the ceiling. The failure is an
    overflow in the sum, not an exponential in the softmax.

**The fix.** Initialise the head explicitly, in fp32, after loading:

```python
torch.nn.init.normal_(model.score.weight, std=0.02)
```

**How to spot it next time.** When you add *any* head to a pretrained model,
assume it is uninitialised until you have checked:

```python
w = model.score.weight
print(w.dtype, w.abs().max().item(), torch.isnan(w).any().item())
```

`NaN` from step one is almost never the data and almost always the
initialisation. And more generally: **`NaN` is a downstream symptom.** Look for
the largest finite value in the network, not for the first `NaN`.

---

## What I would do differently

The through-line across all fourteen: **most of them were not bugs in the
inference engine.** They were bugs in the boundary around it — the container,
the interpreter, the version graph, the harness, the config that was accepted
and ignored.

Four habits that would have caught most of this earlier:

**1. Verify configuration is read, not just accepted.** Three incidents were
silently-ignored settings. `grep` the installed source for the flag name; it
takes seconds and it is decisive.

**2. Ask "which process?" before "what failed?"** Two incidents were interpreter
and subprocess boundaries. `sys.executable` from inside the failing process
answers immediately what experiments answer slowly.

**3. Validate the harness before believing its verdict.** The truncation bug
produced confident, well-formed, wrong conclusions about model quality. Any
measurement apparatus deserves the same scepticism as the thing it measures —
check for truncation, check the template, check the sample count.

**4. Prefer process boundaries when components move fast.** The version deadlock
and the segfaulting trainer integration both dissolved when replaced by an HTTP
boundary. A process boundary is also a dependency boundary and a blast-radius
boundary; that is usually worth more than the efficiency of a shared address
space.

And one worth stating on its own: **`NaN` and segfaults are both downstream
symptoms.** The useful question is what was the last thing that was still
finite, still in-process, still yours.

---

## Where to go deeper

- **Patterns 1–4 (environment failures)** → [25b. Deployment environments](25b-deployment-environments.md) — the full checklist: interpreters, container images, version graphs, subprocess isolation.
- **Patterns 1, 2, 5 (SGLang specifics)** → [26b. SGLang internals](26b-sglang-internals.md) — RadixAttention, default graph capture, model runner subprocess, template mechanics, TP vs DP.
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

## Pattern 1: The knob that does nothing

The worst failures in this list are not crashes. They are settings that are
**accepted, ignored, and never mentioned again.** Nothing errors. The system
starts. It simply does something other than what you asked, and you spend an
afternoon debugging the consequence instead of the cause.

### `VLLM_ATTENTION_BACKEND` is ignored by the V1 engine

**What you see.** You want a specific attention backend — say, to dodge a JIT
failure. You set `VLLM_ATTENTION_BACKEND`. The server comes up. It uses a
different backend anyway.

**What it was.** vLLM's V1 engine does not honour that environment variable.
It is a V0-era control, still present, still settable, no longer wired to
anything on the path that matters.

**Why it happens.** Engines get rewritten. vLLM's V1 rearchitecture changed how
backends are selected, and an env var that used to be load-bearing became
vestigial. Nothing warns you, because from the process's point of view an
unread environment variable is not an error — it is just an environment
variable.

**The fix.** Stop trying to steer the backend that way and remove the cause of
the failure instead (see [pattern 3](#pattern-3-the-container-is-not-the-machine)).

**How to spot it next time.** When a documented knob appears to do nothing,
**grep the installed source for the variable name** before you believe the
docs. `grep -rn VLLM_ATTENTION_BACKEND .venv/lib/python3*/site-packages/vllm/`
answers in seconds what an afternoon of experiments will not: whether anything
reads it at all, and on which code path.

### `enable_thinking` is cosmetic on Qwen3

**What you see.** Qwen3 emits garbled or missing `<think>` tags. You pass
`enable_thinking=False`. Nothing changes.

**What it was.** Two separate problems wearing one costume.

1. Qwen3's chat template **unconditionally injects** `<think>`. The kwarg does
   not gate it; the template does not consult it.
2. SGLang strips the template prefix from the response, so the *opening*
   `<think>` never reaches you — leaving output that looks like a malformed
   thinking block.

**Why it happens.** A chat template is a Jinja program shipped inside the
tokenizer, not part of the serving engine. A kwarg only does something if the
template author wrote a branch for it. "The API accepts this argument" and
"the model's template respects this argument" are unrelated facts.

**The fix.** Prefix `<think>\n` explicitly in the scripts that need it. For the
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

### `SGLANG_EXTERNAL_MODEL_DIR` was deprecated silently

**What you see.** You follow a guide, set `SGLANG_EXTERNAL_MODEL_DIR`, and your
custom model is not found.

**What it was.** Deprecated in SGLang 0.5.6+, replaced by
`SGLANG_EXTERNAL_MODEL_PACKAGE`. The old variable is simply unread.

**Why it happens.** Fast-moving projects rename their extension points. Search
results and blog posts outlive the API they describe, and an unread env var
fails silently by construction.

**The fix.** Use the package variable — though on this project even that did
not survive contact (see
[pattern 2](#pattern-2-which-python-is-installed)).

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

"I installed it" is not a fact about your machine. It is a fact about **one
interpreter**. When another process runs a different interpreter, the package
is not there — and the error message is the same `ModuleNotFoundError` you
would get if you had never installed anything.

### `import vllm` fails inside cluster job scripts

**What you see.** vLLM is installed. It imports fine in your shell. A job
submitted to the cluster dies on `import vllm`.

**What it was.** `uv add` installed into the project's `.venv`. The cluster's
job runner launched scripts with the system interpreter — a base conda
install — which has its own `site-packages` and has never heard of vLLM.

**Why it happens.** There is no global "installed" state on a machine. There is
a set of interpreters, each with a `site-packages`. Activating a virtualenv is
just prepending a directory to `PATH` for *your* shell; a job runner that
invokes an absolute interpreter path never sees it.

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

### An SGLang subprocess cannot import your custom model

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

Some Python packages are not really installed when you install them. They
carry source that gets compiled **the first time you run it**, against
**your** GPU and **your** CUDA toolkit. A runtime image that lacks a compiler
or headers is fine right up to the moment something decides to build.

### FlashInfer's JIT cannot find CUDA headers

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

**The fix.** Two options, both legitimate:

1. **Lock the shapes** so the cached kernels stay valid (TP=4 for one model,
   TP=2 for another) and invoke the entrypoint directly rather than through a
   wrapper that might re-resolve the environment.
2. **Install the headers**: `ninja-build`, `cuda-nvrtc-dev-12-8`,
   `libcublas-dev-12-8` — making the runtime image capable of the compile it
   was always going to attempt.

**How to spot it next time.** If a stack trace mentions `.h` files, `nvcc`,
`ninja` or a build directory, you are looking at a **compile** failure wearing
a runtime failure's clothing. Ask what triggered a build now, and what changed
to invalidate the cache.

### `flash_attn.ops` fails to import on a new GPU generation

**What you see.** vLLM 0.23.0 is installed and refuses to serve;
`import flash_attn.ops` fails.

**What it was.** The PyPI `flash-attn` wheel was not built for B200's
architecture and CUDA version.

**Why it happens.** A wheel is compiled binary. It carries kernels for the
architectures it was built for and an ABI it expects. New silicon arrives
before prebuilt wheels do, and the failure surfaces as a plain `ImportError` —
which reads like a missing package rather than an incompatible binary.

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

Serving stacks pin each other transitively. Once four projects each constrain
the others, "upgrade until it works" can be **provably impossible** — and
recognising impossibility early is worth more than another day of resolver
attempts.

### Ray's LLM serving layer versus the model you need

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

### An RL trainer's vLLM integration segfaults

**What you see.** `trl vllm-serve` segfaults; TRL 1.6 warns about vLLM 0.23
compatibility.

**What it was.** The `GRPOTrainer` ↔ vLLM integration was unstable at those
versions — a C-level crash, not a Python traceback.

**Why it happens.** Two independently-versioned projects sharing a process,
CUDA context and memory pool. When the contract between them shifts, you get a
segfault rather than an error, because the failure is below Python.

**The fix.** Drop the built-in integration and compose explicitly: SGLang
serving rollouts over HTTP, a separate large judge model scoring them, and a
manual training step. More moving parts, each independently debuggable and
independently versioned.

**How to spot it next time.** A **segfault at an integration seam** is rarely
your bug to fix. Check whether the two projects claim to support each other's
versions before debugging further — and prefer a process boundary over a
shared address space when both sides move fast.

!!! note "When the platform is the problem"
    One incident in this project was not technical at all: a managed serving
    submission hung in `PROVISIONING` indefinitely because the required
    namespace was not enabled for the project — an org-admin permission.

    It is here for one reason: **an infinite hang with no error is a signal in
    itself.** Crashes come from code that ran; silent hangs often come from
    code that never started. Check that you are entitled to the resource
    before profiling the thing that is not running.

---

## Pattern 5: Capture demands static everything

[Lecture 13](13-cuda-graphs.md#the-constraint-that-shapes-everything) states the
rule: a captured graph replays **exactly the same operations on exactly the same
memory addresses**, and capture therefore fixes three things — the shapes, the
addresses, and the sequence of commands.

Both incidents here are that constraint being violated. The first violates a
fourth thing the lecture does not list.

### Indexed assignment breaks graph capture

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

### TP=2 crashes during capture

**What you see.** SGLang with `tensor_parallel_size=2` fails with a
cross-device tensor error on B200.

**What it was.** TP > 1 introduces cross-device operations that failed during
graph capture on that platform.

**Why it happens.** TP inserts a collective into every layer — Lecture 22's
all-reduce. Capturing a graph that spans devices means recording operations
whose completion depends on *another GPU*, which multiplies the ways capture
can go wrong.

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

Both of these are a number chosen once, at startup, that becomes a
**correctness** problem much later — and in the second case, a number that
silently corrupted an evaluation.

### `--max-model-len` too small: HTTP 400 on long documents

**What you see.** The server returns 400 on long documents.

**What it was.** Started with `--max-model-len 32768`; the documents exceeded
32K tokens.

**Why it happens.** `max_model_len` is not merely a validation limit — it sizes
the KV cache. From [Lecture 05](05-kv-cache.md#what-it-costs), an engine
reserves cache capacity per sequence, and paged engines carve VRAM into a block
pool up front. Raising the limit means fewer sequences fit concurrently, so the
value is a **real throughput trade**, not a formality.

**The fix.** Restart with `--max-model-len 65536`.

**How to spot it next time.** Measure your **input length distribution** before
choosing the number — the p99, not the mean. And note the honest tension: too
low and long requests are rejected; too high and concurrency drops for every
request. When memory gets tight, engines preempt
([Lecture 09](09-paged-attention.md#preemption)); a limit set far above your
real p99 spends VRAM that could have been batch capacity.

### `max_new_tokens` truncation manufactured a quality failure

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

**The fix.** A hard rule of `max_new_tokens=16384` for evaluation runs.

**How to spot it next time.** Before believing any quality regression, check
whether outputs hit the length ceiling. One line settles it:

```python
truncated = sum(1 for o in outputs if o.finish_reason == "length")
print(f"{truncated}/{len(outputs)} hit the token limit")
```

If that number is not zero, fix the harness before analysing the model.

> This is the single most useful entry on this page for interviews, because it
> is about **trusting your measurement apparatus**. The
> [field notes](field-notes.md) record the same lesson from another operator:
> holding the model fixed and changing *only* the harness swung a long-horizon
> score by up to 18 points. [Lecture 19](19-quantization.md#measuring-what-it-costs)
> makes the matching point about quality metrics — perplexity is necessary and
> not sufficient.

---

## Pattern 7: Uninitialised memory meets low precision

### `loss = 0`, `grad_norm = nan` on a fresh classification head

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

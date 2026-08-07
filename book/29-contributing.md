# 29 — Contributing

**Build:** a merged PR · **Prereq:** [28 — Autoscaling and cost](28-autoscaling-and-cost.md)

---

## Why this is the last lecture

You've built an engine and read the real ones. The remaining gap between you and
someone who does this professionally is mostly **context**: which problems are
open, which tradeoffs are contested, what the maintainers actually worry about.

You get that by participating, not by reading more.

---

## Get the test suite running

Before proposing anything, prove you can build and test the project:

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
pip install -e .                    # slow; a long build
pytest tests/ -x -k "not distributed and not slow"
```

Expect friction — CUDA versions, compilation, hardware-specific skips. Working
through it is part of the job, and it teaches you the project's structure faster
than reading would.

Note which tests skip on your hardware. That tells you what you can and can't
credibly change.

---

## Find real work

**Good first issues.** [vLLM](https://github.com/vllm-project/vllm/labels/good%20first%20issue)
and [SGLang](https://github.com/sgl-project/sglang/labels/good%20first%20issue).
After Parts II–III, scheduler, block-manager, and kernel issues are genuinely
approachable — you've implemented all three.

**Reproduce a bug report.** Underrated and very welcome. Someone reports a
throughput regression; you reproduce it with a minimal case and post a profile.
That's real contribution before you write a line of library code.

**Documentation with numbers.** You now know how to benchmark properly (L25, L26).
Docs backed by a reproducible measurement are rare and valued.

**Your own gap analysis.** Lecture 26 left you with a list of places you're slower
than vLLM. Some of those are places *vLLM* could be faster too. Check.

---

## Reproduce a paper

The strongest single exercise, and you have the substrate for it:

- **[EAGLE](https://arxiv.org/abs/2401.15077)** — better speculation than your
  n-grams (L12). You have the draft/verify loop already.
- **[Medusa](https://arxiv.org/abs/2401.10774)** — multiple decode heads.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)** — better work
  partitioning than your L17 kernel. You'll find out exactly where yours loses.
- **[DistServe](https://arxiv.org/abs/2401.09670)** — disaggregation, against your
  L27 implementation.
- **A KV-cache compression method** — measured with your L19 quality harness.

Pick one, implement it against your engine, and measure. When your numbers differ
from the paper's, work out why — that investigation is where most of the learning
is.

---

## How to be useful

**Bring a measurement.** "This is slow" starts a debate. "This is 2.3× slower than
X, here's the profile and a repro" starts a fix. You have the skills for the
second; use them.

**Scope small.** A focused PR with a test merges. A large refactor stalls.

**Read the discussion first.** Most obvious ideas have been considered. Finding out
*why* it wasn't done is often more informative than the idea.

**Accept that some things are known and unfixed.** Maintainers juggle constraints
you can't see — backwards compatibility, hardware coverage, release timing.

---

## What you know now

Concretely:

- Predict whether an optimization will help, from arithmetic, before building it
- Build the core of a serving engine and explain each design decision
- Write and profile GPU kernels, and know when not to
- Reason about multi-GPU parallelism and its communication costs
- Benchmark honestly, and identify when someone else hasn't
- Trace a performance number to dollars

That's the job.

---

## Keep the habits

Three that outlast the specifics:

**Measure before and after. Always.** Nearly every mistake in this book's making
was caught by running something and reading the output.

**Match the workload to the claim.** Continuous batching looks worthless on
uniform load; speculative decoding looks like magic on code and useless on prose.
A benchmark without a stated workload says nothing.

**Keep the failures.** The kernel that came out slower, the optimization that did
nothing, the prediction that was wrong. Those entries in `notes/` are worth more
than the successes — they're the record of your model of the machine being
corrected.

---

## Go deeper

- **Kiely Appendix B** (p.231–256) — curated reading by topic.
- **vLLM `CONTRIBUTING.md`** and the RFC process for larger changes.
- **[vLLM Slack / SGLang Discord]** — where design discussion actually happens.
- **[Appendix B](appendix-b-reading.md)** — this book's papers, indexed to
  lectures.

---

**That's the book.** Go make something fast.

# Repo conventions

A learning repo. The goal is **understanding**, not shipping — which changes what
"good" means here.

## The prime directive: measure first

No optimization lands without a before/after number from `bench/`. If a change
doesn't move a metric, we don't know it worked — and "it should be faster" is not
a result.

Order of operations, always:

1. Measure the baseline
2. Write down the prediction (in the note, before running)
3. Build the change
4. Measure again
5. Record what moved — **and what surprised you**

## Don't do the work for me

This is a Zero-to-Hero-style course. When I'm on a milestone:

- **Don't implement the milestone unless I ask.** Nudge toward the insight instead.
- If I'm stuck, ask what I've tried and what I measured before offering code.
- Reviewing my implementation, questioning my approach, and explaining a concept
  are all fair game — writing the milestone for me defeats the point.
- Scaffolding, plumbing, plotting, and debugging help are fine.

## Benchmarking rules

- **Always `synchronize()` before and after timing.** GPU work is async;
  unsynchronized timing measures kernel *enqueue*, not execution. `bench/harness.py`
  handles this — use the harness rather than hand-rolling `time.perf_counter()`.
- **Report p50/p90/p99, never the mean alone.** The tail is what users feel.
- **Match the workload to the optimization.** `bench/workloads.py` documents which
  workload exposes which win. Continuous batching looks like nothing on `uniform`
  and shines on `mixed_length`; that mismatch is a classic way to fool yourself.
- Commit result JSONs in `bench/results/`. They're the record.
- Same model (`Qwen3-0.6B`) everywhere so numbers stay comparable.

## Capture every good question — automatically

**Standing rule: when a question in conversation produces an explanation that
clarifies something, write it into `book/qa.md` before moving on.** Do not wait
to be asked. This has already caught several real errors in the lectures.

Two places, both required:

1. **`book/qa.md`** — the full answer, under a `##` heading phrased as the
   question that was actually asked. Include the wrong intuition and why it's
   wrong; the misunderstanding is the valuable part.
2. **The relevant lecture** — a `??? question "..."` collapsible aside at the
   exact point a reader would stall, with a 2–3 sentence summary and a link to
   the Q&A entry. Forward links matter more than backward ones: the reader is in
   the lecture, not the appendix.

Prefer questions that came from genuine confusion over ones I invented. If an
answer contradicts the book, **fix the book too** and say so in the commit.

Then rebuild: `uv run mkdocs build --strict` (must be warning-free).

## Book math — derive everything, condense nothing

**Standing rule: every math step in the book gets its full derivation, always.**
Do not skip, collapse, or "leave as an exercise" any intermediate step. A
formula like `8N² + 8Nd` is the *conclusion* of a derivation, not a fact to
state. Show the traffic per operation, the term-by-term substitution, the
factorisation, and the numbers plugged in. Condensed math is the fastest way to
lose the reader the lecture was written for.

## Notes

`notes/` is the deliverable, not an afterthought — prose and runnable code
interleaved, Karpathy-style. Every code block should actually run; if a snippet
drifts from the code, the note is wrong.

Keep failures and wrong predictions **unedited**. The kernel that came out slower
and the optimization that did nothing are the highest-value entries in the repo.

## Code style

- Readability over cleverness — this code is meant to be *read*, including by me
  in six months.
- Comment the *why*, especially where the non-obvious choice is deliberate.
- Prefer explicit over magic; this is the one codebase where a little verbosity is
  the point.
- Stubs carry a docstring naming their milestone and the number that must move.

## Hardware

Laptop is Apple M1 / 8GB — fine for Phase 0 and M1.1–M1.4. From M1.5 on, work
runs on a rented NVIDIA box.

Write code locally, batch GPU sessions, **always stop the pod**. When suggesting a
GPU workflow, prefer things that can be prepared offline and executed in one run.

# Inference Zero-to-Hero

*Building an LLM inference engine from scratch, measuring everything.*

You build KV caching, continuous batching, paged attention, prefix caching and
speculative decoding, then read vLLM and understand why it's built the way it is.

[Start reading :material-arrow-right:](00-intro.md){ .md-button .md-button--primary }
[The repo :fontawesome-brands-github:](https://github.com/prabodh1194/learn-inference){ .md-button }

!!! quote "The method"
    **build the naive thing → measure it → find the bottleneck → fix it → measure again**

    You write a slow generation loop on purpose, plot how badly it scales, and
    only *then* build the KV cache. Reading "the KV cache avoids recomputation"
    teaches you a sentence; watching your own per-token latency climb and then
    flatten teaches you the thing.

---

## How to read this

Each lecture gives you four things:

| | |
|---|---|
| **the text** | the idea, derived, 5–10 minutes |
| **a demo** | `uv run python book/code/NN_*.py`, shows the phenomenon |
| **a build** | you implement it in `engine/` |
| **a test** | `pytest tests/test_NN_*.py`, proves it's right |

And a **number that must move**. Record it in `notes/`.

**Run the demo before reading "The idea."** Seeing the waste as a number first is
what makes the fix feel inevitable rather than arbitrary.

!!! warning "Don't skip the predictions"
    Several lectures ask you to guess before measuring. Being wrong in writing is
    how the intuition forms; skipping to the answer feels efficient and teaches
    much less.

---

## Three ways in

<div class="grid cards" markdown>

-   :material-numeric-1-box: **Follow the path**

    ---

    Numbered steps 1–41, setup through landing a PR.

    [START-HERE](https://github.com/prabodh1194/learn-inference/blob/main/START-HERE.md)

-   :material-book-open-variant: **Read straight through**

    ---

    Intro, then Part I. Demos run with no GPU.

    [00. Introduction](00-intro.md)

-   :material-help-circle: **Chase a confusion**

    ---

    Worked answers to real questions, several of which found book errors.

    [Q&A](qa.md)

</div>

---

## Field notes

[**field-notes.md**](field-notes.md) collects what practitioners report, real
magnitudes from real deployments, and the places where an optimization
disappointed someone. Books give you the mechanism; these give you the scale.

Use them to sanity-check your own results, and add your own entries as you go.

---

## Companion texts

This book indexes two excellent sources rather than duplicating them:

- **Philip Kiely, *Inference Engineering*** (Baseten, 2026): the breadth-first
  survey. Lectures cite it by section; read those sections when pointed at them.
- **Aleksa Gordić, [*Inside vLLM*](https://www.aleksagordic.com/blog/vllm)**, a
  top-down read of vLLM V1. **Save it for Lecture 14.** Read before you've built
  a scheduler and it teaches vocabulary; read after and it teaches judgment.

---

## Status

**All 30 lectures are written**, front matter through appendices.

The **code** is a different matter, and deliberately so: `engine/`, `kernels/`,
`jaxlm/`, and `serve/` are stubs for *you* to fill in. The demos in `book/code/`
run today; the tests in `tests/` fail until you build each lecture. That failure
list is the course.

Run `uv run python scripts/progress.py` to see where you are.

# 24b. Serving agents

**Build:** `engine/agent_context.py` · **Test:** `tests/test_24b_agent.py`
**Moves:** tokens per completed task — context growth is the cost agent serving optimizes
**Prereq:** [24. Serving](24-serving.md)

---

## The problem

The dominant LLM workload in 2026 is not one request, it's an **agent**: a loop
of model calls interleaved with tool calls. Each round-trip appends a prompt,
a tool result, and an action to the conversation, so the context grows
*linearly with the number of steps* — and context is exactly the resource
Lecture 05 showed you is finite and Lecture 09 showed you how to budget.

An agent that runs 2,000 steps is "tens of millions of tokens in context". The
book's ~8k crossover (Lecture 05) is crossed within the first ~100 steps. Agent
serving is long-context serving, and the optimization is not kernels — it's
*deciding what stays in the context*.

---

## The idea

### Context is the scarce resource

Every other resource has a manager in this book: the block pool manages KV
memory (L09), the scheduler manages slots (L08). Agents need a manager for the
*context itself*. The production pattern is a **lazy escalation ladder** — the
cheapest, least destructive move first, escalate only when forced:

1. **trim** — drop the oldest turns whole (they're the least relevant)
2. **snip** — truncate a turn's text, keep its head
3. **micro-compact** — compress the middle turns into short summaries
4. **context collapse** — summarize the entire prior conversation into a few
   lines and start fresh
5. **auto-compact** — the harness does all of the above automatically when the
   budget is hit

The ordering matters because each rung destroys more information. Trimming old
turns loses almost nothing; collapsing the whole conversation loses a lot. You
climb one rung at a time, and only as far as the budget forces.

### Tool definitions are prompt tokens

Every tool you hand the model — its name, description, and input schema — is
serialized into the prompt, so it costs **KV cache**, permanently, for the whole
session. A hundred tools is a hundred permanent prefill tokens before the model
has done anything. The fix is **deferred schemas**: ship a name-only menu, and
load a tool's full schema only when the model actually reaches for that tool.
This is the same "don't pay for what you don't use" instinct as prefix caching
(L10), applied to the prompt itself.

### Subagents isolate context

The complementary technique: instead of one long context, spawn a **subagent**
with its own fresh context window, give it a narrow task, and take back only a
short summary. Isolation is the point — the subagent's dead ends and digressions
don't pollute the parent's context. It costs, though: a reported ~7× the tokens
of a normal session, because the subagent re-derives context the parent already
had. **Delegation is context hygiene at a price**, and the price is real.

### Scoring is cheap; generating is dear

An agent that picks among candidate actions is doing **best-of-N**: generate
several candidates, score them, keep the best. The two halves sit on opposite
sides of your roofline (L02):

- **Scoring** N finished candidates = one *prefill*, compute-bound, all tokens
  at once.
- **Generating** N candidates = N *decodes*, memory-bound, weights re-read per
  token.

So the "think of five plans, pick one" pattern is cheap on the picking side and
dear on the generating side — which is why you generate few and score cheaply,
never the reverse.

---

## The code

```python
def compaction_ladder(turns, budget):
    # turns: list of (role, text), oldest first
    # returns (final_token_count, actions)
    actions = []
    while token_count(turns) > budget and turns:
        oldest = turns[0]
        turns = turns[1:]                     # rung 1: trim the oldest
        actions.append(("trim", tokens(oldest)))
    return token_count(turns), actions
```

That is the cheapest rung. The full ladder adds the summary rungs on top of it,
each replacing several turns with one shorter summary instead of deleting them.

---

## Build it

1. Implement `compaction_ladder` in `engine/agent_context.py`: trim first, and
   add the `snip` and `compact` rungs (a `compact` replaces N turns with one
   summary of a fixed token count).
2. `uv run pytest tests/test_24b_agent.py -v`. The key assertion: the newest
   turn always survives, and the budget is always met.
3. Simulate a 500-step agent with 100 tokens per round-trip and watch the
   context line: 50,000 tokens uncompressed (500 × 100). Run the ladder at a
   8,192 budget and report how many turns survive.
4. (Optional) Instrument your own agent loops, if you run any, with the number
   of tokens per round-trip. It is the single most informative number about why
   an agent is slow.

---

## What you should see

**Context grows linearly with steps.** No surprise — but the *slope* is your
real cost per step.

**The ladder buys back the budget cheapest-first**, and the newest turn is
inviolable — that's the working memory you must never drop.

**Subagents and deferred schemas are trades, not wins.** Each has a documented
cost (7× tokens, or a schema-load round-trip), and the field notes record them
because people keep underestimating it.

---

## Go deeper

- **[Field notes](field-notes.md)**: the compaction ladder, deferred schemas, and
  the "1.5% model, 98% plumbing" breakdown of a production agent.
- **[Best-of-N sampling](https://arxiv.org/abs/2407.16603)** and the
  `log n − (n−1)/n` KL bound on how much reranking can drift from the base model.
- **Kiely §7** (p.183+): agent and tool-calling workloads at the serving layer.
- **MCP (Model Context Protocol)**: the transport standard for tools; the
  per-model JSON-schema formats sit on top of it (Lecture 12b).

---

## Check yourself

1. Why does an agent's context grow *linearly* with steps, and why is that the
   resource to manage rather than the GPU?
2. What is the cheapest, least destructive rung of the compaction ladder, and
   why does it come first?
3. Why do tool definitions cost KV cache, and how does a deferred schema avoid
   paying it?
4. A subagent isolates context at ~7× the tokens. When is that a good trade?
5. In best-of-N, which half is prefill-bound and which is decode-bound, and why
   does that decide how many candidates you generate?

---

## Next

**[25. Load testing](25-load-testing.md)**: agents are the workload that makes
queues and tail latency bite — many small dependent calls instead of one long
one.

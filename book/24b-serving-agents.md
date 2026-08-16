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

Put numbers on "grows linearly", because the slope is the whole story. Say each
round-trip adds ~100 tokens (a prompt, a tool result, an action — the figure you
will measure yourself in the build section):

```
step   10  :     10 × 100  =    1,000 tokens
step  100  :    100 × 100  =   10,000 tokens   <- past L05's ~8k crossover
step  500  :    500 × 100  =   50,000 tokens
step 2,000 :  2,000 × 100  =  200,000 tokens   <- a frontier context window, full
```

So the book's ~8k KV crossover (Lecture 05) is passed inside the first ~100
steps, and a long-running agent walks a full 200k context window by step 2,000.
Heavier round-trips — a large file read, a verbose tool result — push all of
this earlier; agents that paste whole documents into context reach the same
place in tens of steps rather than hundreds.

Agent serving is therefore long-context serving, and the optimization is not
kernels — it's *deciding what stays in the context*.

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

Seen against the growing context, the ladder is a sawtooth — growth, a cut,
growth again:

```
 tokens
        │                              ╱│              ╱│
 budget ├──────────────╱│─────────────╱ │────────────╱ │───  the budget line
        │            ╱  │           ╱   │          ╱   │
        │          ╱    │         ╱     │        ╱     │
        │        ╱      ▼       ╱       ▼      ╱       ▼
        │      ╱      trim    ╱        snip  ╱      compact
        │    ╱                                (rung 3)
        │  ╱   ← context grows ~100 tokens per step
        └──────────────────────────────────────────────────>  steps

        each ▼ is a rung firing; the rung climbs only when
        the cheaper one below it can no longer buy enough room
```

**And here is the cost the diagram hides.** Every one of those cuts rewrites the
*beginning* of the context — trimming drops the oldest turns, compaction
replaces them with a summary. Prefix caching (L10) keys on an exact token
prefix, so the moment you change the front of the prompt, **every cached block
after the change is invalidated** and the next call re-prefills the whole
conversation from scratch.

That reframes the ladder entirely. Its cost is not the summarizing model call
(though micro-compaction and collapse are real model calls, and are not free) —
it is the lost prefix cache on the step after the cut:

```
before a cut:  [ system │ tools │ turn 1 │ ... │ turn N ]
               └────────── all cached, prefill ≈ free ─────┘

after a cut :  [ system │ tools │ SUMMARY │ turn N ]
               └─ cached ─┘└──── changed: re-prefill everything after here ────┘
```

Which gives the real design rule: **compact rarely and in big steps.** Ten small
trims cost ten cache invalidations; one larger compaction costs one. It also
explains why deferred tool schemas (next section) are subtler than they look —
loading a schema mid-session changes the prompt prefix too, with exactly the
same effect.

??? question "Wait — compaction makes the context smaller. How can it cost more?"
    It makes every *future* step cheaper and the very next step much more
    expensive. Prefix caching matches an exact token prefix, so rewriting the
    front of the context invalidates every cached block after the change: the
    next call re-prefills the whole surviving conversation from scratch. You
    shrank the ongoing cost and paid a one-off bill for it — which is why
    frequent small trims are the intuitive policy and the expensive one.
    [Full answer](qa.md#why-does-compaction-hurt-the-prefix-cache)

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

??? question "If a subagent costs ~7× the tokens, why would anyone use one?"
    Because tokens spent and context *kept* are different currencies. The
    subagent re-derives what the parent already knew (that's the 7×), but hands
    back only a short summary — so the parent's context grows by hundreds of
    tokens instead of tens of thousands. It's a good trade when the exploration
    is long and mostly discardable, and a bad one when the task is short or the
    parent actually needs the details rather than the conclusion.
    [Full answer](qa.md#if-a-subagent-costs-7-the-tokens-why-use-one)

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
    # len(turns) > 1, NOT just `turns`: stop while one turn remains, so the
    # newest turn can never be trimmed. That is the invariant the tests check.
    while token_count(turns) > budget and len(turns) > 1:
        oldest = turns[0]
        turns = turns[1:]                     # rung 1: trim the oldest
        actions.append(("trim", tokens(oldest)))
    return token_count(turns), actions
```

Look closely at the loop condition, because the obvious version is wrong.
Writing `while token_count(turns) > budget and turns:` reads fine and passes
casual inspection — but if a single turn is itself bigger than the budget, it
deletes turns until the list is *empty*, taking the newest turn with it. The
agent then has no working memory at all and the next model call has nothing to
act on.

`len(turns) > 1` stops one turn early. The consequence is worth stating plainly:
**the ladder can return a context that still exceeds the budget.** That is the
correct behaviour — one over-budget turn is a problem you solve by *snipping*
that turn (rung 2), not by deleting the only thing the agent knows. A rung-1-only
implementation should hand that case upward rather than pretend it succeeded.

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

# 24. Serving

**Build:** `serve/api.py` · **Test:** `tests/test_24_serving.py`
**Moves:** nothing about the engine, everything about whether it's usable
**Prereq:** [23. MoE and expert parallelism](23-moe-and-expert-parallelism.md)

---

## The problem

You have an engine. It's a Python function.

A service is different in ways that matter: requests arrive concurrently over the
network, users expect tokens to stream as they're produced, clients disconnect
mid-generation, and one slow request must not stall the others.

None of that is engine work. All of it determines whether the engine is usable.

---

## The idea

### Speak OpenAI's API

Not because it's well-designed, but because it's the de facto standard: the
format every existing client and tool already speaks. Implement
`/v1/chat/completions` and every existing client works unmodified: that's real
leverage for free.

```
POST /v1/chat/completions
{
  "model": "qwen3-0.6b",
  "messages": [{"role": "user", "content": "Why is decode memory-bound?"}],
  "stream": true,
  "max_tokens": 256,
  "temperature": 0.7
}
```

### Streaming is the point

Lecture 01's TTFT (time to first token — the gap between sending the request and
the first token arriving) only matters if the user *sees* the first token. Buffer
the whole response and you've thrown away every latency optimization in this book:
the user
waits for the entire generation and receives it all at once, so a fast engine
feels like a slow one.

**Server-Sent Events (SSE)** is a plain-text protocol: the server opens one HTTP
connection and pushes lines of `data:` text as each one becomes ready. Each token
is a line, so the client renders it the moment the GPU produces it. One chunk per
token:

```
data: {"choices":[{"delta":{"content":"Because"}}]}

data: {"choices":[{"delta":{"content":" decode"}}]}

data: [DONE]
```

### Decouple the API server from the engine loop

The architectural point of this lecture, and the reason vLLM runs the API server
in a **separate process**: the objective here is to see why that separation is
necessary, not a preference.

An **event loop** is a single thread that handles every event a program receives,
one at a time: one request arriving, then another socket becoming readable, then
a timer. Everything that happens in that thread happens in sequence, so any
unrelated work it does delays everything else. Two very different kinds of work
are competing for one thread:

- The **engine loop** is synchronous and GPU-bound: one step at a time, back to
  back, with no gaps. It should never wait on a socket, because a socket has
  nothing to do with the GPU's work.
- **HTTP handling** is async and I/O-bound: parsing a request, serializing JSON,
  and writing to a slow client each involve waiting, on a network, on a client
  halfway around the world, for data that hasn't arrived yet.

Put them in one event loop and request parsing, JSON serialization, and SSE
writes all steal time from the scheduler (the engine's step loop, Lecture 08). Picture a chef who also answers the
phone: every call that comes in while the stove is on makes the stove wait too,
because there's only one person. The fix is a second process with its own event
loop, connected by queues, the serving equivalent of a waiter taking orders:

```
BAD:   [async HTTP] ---- shares event loop ---- [engine.step()]
       every request handler delays the next decode step

GOOD:  [API server] --queue--> [engine loop] --queue--> [API server]
       the engine never waits on a socket
```

Tokenization is the sneaky one: it's pure CPU, it's cheap per call, and it sits
on the **request path**, the stretch of work between "request arrives" and
"first token streams". At high request rates it will eat your scheduler's time
if it shares the loop.

??? question "Why can't the API server just be a thread in the same process?"
    Python threads share one interpreter lock (the GIL), so their code never
    runs at the same time as the engine loop's: every byte of HTTP work would
    still execute between engine steps, just as if it were the same thread. A
    separate process has its own interpreter and its own event loop, so HTTP
    work and engine steps genuinely run in parallel, and the engine's only
    contact with the outside world is the queues.
    [Full answer](qa.md#why-cant-the-api-server-just-be-a-thread-in-the-same-process)

### Cancellation

Clients disconnect, closed tabs, timeouts, abandoned agent runs. If you don't
detect it you keep generating tokens nobody will read, burning GPU on nothing.
It's the phone call where the other person hung up minutes ago and nobody
mentioned it to you.

Under load this is not a rounding error. Check for disconnection each step and
free the sequence (one request's in-flight generation), releasing its blocks
(Lecture 09) back to the pool.

### The whole path, and where time goes

The objective here is to have a map of the whole path, so a measured delay has a
place to look.

```
   once, at the start                 then, once PER TOKEN
   ┌──────────────────────────┐   ┌──────────────────────────────────┐
   tokenize → queue → schedule → prefill → ┌─────────────────────────┐
                                           │ decode                  │
                                           │   ↓                     │
                                           │ detokenize (incremental)│ × N
                                           │   ↓                     │
                                           │ SSE flush ──► client    │
                                           └─────────────────────────┘
```

**The loop is the important part.** Detokenize and SSE are *inside* it, not
after it. Each decode step produces one token, which is turned into text and
pushed to the client immediately — that is what makes the response stream. Put
detokenize and SSE after the loop instead and you have rebuilt the buffered
server from the top of this lecture: same total time, no visible output until
the end.

The arrows are handoffs between components. The queue is where requests wait;
schedule is the scheduler's choice of which sequences run this step; prefill and
decode are the engine steps from Lecture 01; and the SSE stream is what the
client actually sees.

??? warning "Incremental detokenization is not `decode(token)`"
    Detokenizing one token at a time is the part people get wrong, and it fails
    in a way that looks like a model bug.

    A BPE token is not a character, and it is not even necessarily a whole
    character. Multi-byte UTF-8 — emoji, CJK, accented letters — is routinely
    split across two tokens. Decode each token in isolation and the first half
    has no valid character in it:

    ```
    token 4712  ──► b'\xf0\x9f'      decode alone → "\ufffd"  (garbage)
    token 9931  ──► b'\x91\x8b'      decode alone → "\ufffd"  (garbage)
                    ────────────
                    together    ──►  b'\xf0\x9f\x91\x8b'  =  "👋"
    ```

    Tokenizers also merge leading spaces into tokens, so naive per-token decode
    produces wrong spacing even on pure ASCII.

    The fix every engine uses: keep a small window of recent token IDs, decode
    the window, and emit only the *new suffix* — the characters that appeared
    since the last step. Bytes that do not yet form a complete character stay
    buffered until the next token completes them. `transformers` exposes this
    as an incremental detokenizer; vLLM has its own
    (`detokenize_incrementally`) for exactly this reason.

    Test it with an emoji in the output. A naive implementation shows `\ufffd`
    where the emoji should be.

Kiely §7.5.1 (p.205) makes a point worth taking seriously: **client-side
overhead can dominate**. A 20ms TTFT is invisible behind a 200ms TLS handshake
(the encrypted negotiation every HTTPS connection performs before the first byte
flows), a slow tokenizer, or a client that buffers. Measure end to end, from the
client, or you're optimizing a number nobody experiences.

---

## Build it

1. FastAPI app in `serve/api.py` with `/v1/chat/completions` (streaming and
   non-streaming) and `/v1/models`.
2. Run the engine loop in a **separate process**, communicating over queues.
3. Apply the model's chat template, `tokenizer.apply_chat_template`. Getting this
   wrong produces subtly worse output that looks like a model problem.
4. **Detokenize incrementally**, not one token at a time in isolation. Keep a
   window of recent token IDs, decode it, emit only the new suffix, and hold
   back bytes that don't yet complete a character. Verify with a prompt that
   makes the model emit an emoji or CJK text — naive per-token decode shows
   `\ufffd` there and nowhere else, which is why it survives casual testing.
5. Handle disconnection; free the sequence promptly.
6. `uv run pytest tests/test_24_serving.py -v`
7. Point a real client at it:

```bash
curl -N localhost:8000/v1/chat/completions \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Then try the `openai` Python client against your server. It should just work.

8. **Measure client-observed TTFT** and compare to your engine-internal TTFT. The
   gap is your serving overhead, and it's the number your users actually feel.

---

## What you should see

**Client-observed TTFT higher than engine TTFT.** Always. HTTP, tokenization,
serialization. Small is fine; large means something is wrong.

**Streaming feels dramatically better** at identical total latency. Perceived
performance is a real thing.

**Throughput drops if the API server shares the engine's event loop.** Try it
deliberately: it's a convincing demonstration.

---

## Go deeper

- **Kiely §7.5–7.5.3** (p.204–207), client overhead, async inference, streaming
  protocols.
- **Kiely §7.1** (p.179), containerization and dependency management.
- **vLLM `vllm/entrypoints/openai/api_server.py`**: the production surface. Note
  how much is spec compatibility rather than inference.
- **[OpenAI API reference](https://platform.openai.com/docs/api-reference/chat)**:   the shape you're implementing.

---

## Check yourself

1. Why does vLLM run the API server in a separate process?
2. Streaming doesn't change total generation time. Why does it matter?
3. A client disconnects at token 5 of 500. What should happen, and what does it
   cost if nothing does?
4. Engine TTFT is 45ms; client TTFT is 380ms. Where would you look first?
5. Why is tokenization specifically dangerous on the engine's event loop?

---

## Next

**[24b. Serving agents](24b-serving-agents.md)**: the dominant workload on that
service — tool loops, context growth, and the compaction ladder.

**[25. Load testing](25-load-testing.md)**: find out what your service actually
does under pressure.

**Use an open loop.** A closed-loop test, N clients each waiting for a response
self-limits, so it literally cannot show you overload. Sweep the arrival rate
and find your knee.

**Predict your knee before measuring.** Most people guess high.

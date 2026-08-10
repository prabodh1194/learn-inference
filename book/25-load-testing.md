# 25 — Load testing

**Build:** `bench/load_test.py` · **Test:** `tests/test_25_load.py`
**Moves:** your understanding of what your service can actually promise
**Prereq:** [24 — Serving](24-serving.md)

---

## The problem

Every number so far came from a benchmark you controlled — fixed batches, requests
you fired all at once. Real traffic doesn't behave that way.

Requests arrive **randomly**. They **queue**. Latency depends on what else is
running. And the honest answer to "how fast is your service?" is not a number —
it's a **curve**.

---

## The idea

### Poisson arrivals, not a fixed loop

Firing 100 requests simultaneously measures a burst. Real arrivals are random with
some average rate, which means occasional clumps — and clumps are what create
queues, and queues are what create tail latency.

`bench/workloads.py::poisson_arrivals` stamps exponentially-distributed gaps onto
any workload:

```python
from bench.workloads import mixed_length, poisson_arrivals
load = poisson_arrivals(mixed_length(n=1000), rate=20.0)   # 20 req/s
```

Using a fixed inter-arrival time hides queueing entirely and makes your service
look far more predictable than it is.

### Open vs. closed loop

A distinction that changes what you're measuring:

**Closed loop** — N clients, each waits for its response before sending again.
Load *self-limits*: if the server slows down, offered load drops. This is what
most naive load tests do, and it cannot show you overload.

**Open loop** — requests arrive at a fixed rate regardless of how the server is
doing. If service is slower than arrival, the queue grows without bound.

**You want open loop.** It's what real traffic does, and it's the only way to find
where your service falls over.

### The knee

Sweep arrival rate and plot latency against throughput:

```
throughput ^
           |        ,-------------  <- saturated
           |      ,'
           |    ,'  <- the knee
           |  ,'
           |,'
           +------------------------> offered load

latency    ^                    /
           |                   /  <- p99 runs away
           |          ________/
           |_________/
           +------------------------> offered load
```

Below the knee, throughput rises with load and latency is stable. Above it,
throughput is flat — you're saturated — and **latency grows without bound** because
the queue does.

**The knee is your operating limit.** Run above it and latency depends on how long
you've been overloaded, which is not a service you can make promises about.

This is also the honest answer to "how many requests per second can you handle?"
It's "X at p99 under Y ms," and any single number without a latency bound is
marketing.

### What to report

Per load level: offered rate, achieved throughput, TTFT p50/p90/p99, TPOT
p50/p90/p99, end-to-end p50/p90/p99, and **queue depth over time**.

Queue depth is the diagnostic. Growing monotonically means you're past the knee,
and it tells you *before* the latency numbers do.

### Percentiles need samples

Lecture 04's caveat, now load-bearing. A p99 from 100 requests interpolates
between your two slowest samples — nearly meaningless. **Thousands of requests**
before you quote a p99, and report the sample count alongside it.

---

## Build it

1. Write `bench/load_test.py`: open-loop driver, Poisson arrivals, configurable
   rate, concurrent HTTP against your Lecture 24 server.
2. Sweep offered load from well under capacity to well over.
3. Plot with `bench/plot.py::latency_throughput`.
4. **Find your knee.** Record the rate, and p99 at that rate.
5. Run the same sweep against **real vLLM** on the same hardware and workload.
   That's Lecture 26.
6. Compare your driver against vLLM's `benchmarks/benchmark_serving.py` — a
   sanity check that you're measuring the same thing.

**Predict first:** what request rate do you think is your knee? Write it down.
Most people guess high.

---

## What you should see

**A clear knee**, often at lower load than you expected.

**p99 rising much faster than p50** as you approach it. The tail always goes
first — which is why Lecture 04 insisted on percentiles.

**Queue depth growing without bound** past the knee. This is the unambiguous
saturation signal.

**Different knees for different workloads.** `long_prefill` saturates earlier than
`mixed_length`. Your capacity is a property of *traffic shape*, not just hardware.

---

## Go deeper

- **Kiely §4.5–4.5.2** (p.112–114) — benchmarking tooling and practical tips.
- **Kiely §1.4.2** (p.37) — end-to-end metrics, including what the client sees.
- **vLLM `benchmarks/benchmark_serving.py`** — the standard harness. Note how much
  is realistic load generation rather than timing.
- **[Open Versus Closed: A Cautionary Tale](https://www.usenix.org/legacy/event/nsdi06/tech/schroeder.html)**
  (Schroeder et al., NSDI '06) — why the distinction above changes your
  conclusions. Not LLM-specific and directly applicable.
- **[Systems Performance](https://www.brendangregg.com/systems-performance-2nd-edition-book.html)**
  (Gregg) — the USE method and latency analysis generally.

---

## Check yourself

1. Why does an open-loop test find problems a closed-loop test can't?
2. Past the knee, throughput is flat but latency grows. Where is the time going?
3. Why is "we handle 500 req/s" meaningless without a latency bound?
4. Why does `long_prefill` saturate earlier than `mixed_length`? *(Lecture 11.)*
5. Your p99 came from 200 requests. What's wrong with quoting it?

---

## Next

**[26 — Versus vLLM](26-versus-vllm.md)** — benchmark against the real thing.

**You will lose. That is the expected result.** The deliverable is not a win —
it's a per-subsystem explanation of the gap, with profiler numbers.

It also turns your skepticism outward: after this, you'll know exactly which
five questions to ask about anyone else's benchmark.

# 25. Load testing

**Build:** `bench/load_test.py` · **Test:** `tests/test_25_load.py`
**Moves:** your understanding of what your service can actually promise
**Prereq:** [24. Serving](24-serving.md)

---

## The problem

Every number so far came from a benchmark you controlled, fixed batches, requests
you fired all at once. Real traffic doesn't behave that way.

Requests arrive **randomly**. They **queue**. Latency depends on what else is
running. And the honest answer to "how fast is your service?" is not a number,
it's a **curve**.

---

## The idea

### Poisson arrivals, not a fixed loop

Firing 100 requests simultaneously measures a burst. Real arrivals are random with
some average rate, which means occasional clumps, and clumps are what create
queues, and queues are what create tail latency.

"Poisson arrivals" is the statistician's name for traffic like that: a fixed
average rate, but every gap between arrivals is random, so requests bunch up
instead of arriving like clockwork. At 20 req/s the gaps average out to
`1/20 = 0.05 s`, 50 ms, but individual gaps swing from a few milliseconds to a
few hundred, and where several short gaps land in a row, requests pile on top of
each other. That clump is real load, and a test with fixed spacing never
produces one. The `exponentially-distributed gaps` in the code below are exactly
this: each wait is drawn at random from a distribution whose average is the
1/rate you asked for.

`bench/workloads.py::poisson_arrivals` stamps exponentially-distributed gaps onto
any workload:

```python
from bench.workloads import mixed_length, poisson_arrivals
load = poisson_arrivals(mixed_length(n=1000), rate=20.0)   # 20 req/s
```

Using a fixed inter-arrival time hides queueing entirely and makes your service
look far more predictable than it is.

??? question "Where does the per-step cadence come from, and is it always ~25 ms?"
    Decode steps don't take a variable amount of time; each one is a roughly
    fixed wall-clock beat, set by the bytes the step must read divided by
    bandwidth. **How long that beat is depends entirely on how full the card
    is**, so there is no single universal number — and the often-quoted ~25 ms
    is the *saturated* case, not yours.

    The upper bound is reading the card's entire memory, `capacity / bandwidth`:
    on a 3090 that's `24 GB / 936 GB/s ≈ 26 ms`. That's the beat for a model
    whose weights plus KV cache fill the GPU. Our Qwen3-0.6B does not:

    ```
    saturated card :  24 GB    / 936 GB/s  =  25.6 ms   <- the "~25 ms" figure
    Qwen3-0.6B     :  880.8 MB / 936 GB/s  =   0.94 ms  <- this book's model
    ```

    A ~27× difference, so don't carry the 25 ms into your own measurements —
    derive your own from what your step actually reads (weights, plus the KV
    cache of everything in the batch). The *structure* of the argument survives
    at any cadence: a request arriving just after a step's batch is fixed waits
    up to one full cadence to *board* the next step, then another for that step
    to finish, giving a **two-cadence worst-case floor** under any queueing, at
    any load. On a saturated card that's ~50 ms; on ours it's ~2 ms. The train
    leaves on a schedule and you wait for it even when it's empty — but check
    the timetable before quoting the wait.
    [Full answer](qa.md#where-does-the-25-ms-per-step-cadence-come-from)

??? question "Arrivals average 20 req/s and the server keeps up on average. Why does a queue ever form?"
    Because "on average" describes no particular moment. The 50 ms mean gap
    hides bursts: several requests can land within milliseconds of each other,
    while the server can only chew through so many per instant. The burst
    exceeds that instant's capacity, so some requests wait, and the queue drains
    only when a quiet gap arrives. The closer average arrival sits to capacity,
    the longer and deeper those lines get, and past the knee they never drain,
    which is exactly what the open-loop test is built to expose.
    [Full answer](qa.md#arrivals-average-20-reqs-and-the-server-keeps-up-on-average-why-does-a-queue-ever-form)

### Open vs. closed loop

A distinction that changes what you're measuring. Think of a restaurant door:

```
closed loop:  client -> request -> server -> response -> next request
              nobody sends a new request while an old one is in flight

open loop:    --- requests arrive on schedule, come what may ---
              the next request is sent on time whether or not the
              previous response has come back
```

**Closed loop**: N clients, each waits for its response before sending again.
That's the maitre d' who only admits a new table when someone leaves: if the
kitchen slows down, tables free up more slowly, fewer people get let in, and the
pressure on the kitchen drops by itself. Load *self-limits*: if the server slows
down, offered load drops. This is what most naive load tests do, and it cannot
show you overload, because the test supplies less and less pressure the worse
the server copes.

**Open loop**: requests arrive at a fixed rate regardless of how the server is
doing, the entrance that admits strangers on a schedule no matter how the crowd
inside is going. If service is slower than arrival, the queue grows without
bound, and the test keeps pouring arrivals in until you stop it.

**You want open loop.** It's what real traffic does, and it's the only way to find
where your service falls over.

### The knee

Sweep arrival rate and plot latency against throughput. **Offered load** is the
rate at which you push requests at the server, whether or not it can keep up;
it goes on the horizontal axis:

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
throughput is flat (you're saturated) and **latency grows without bound** because
the queue does.

**The knee is your operating limit.** Run above it and latency depends on how long
you've been overloaded, which is not a service you can make promises about.

This is also the honest answer to "how many requests per second can you handle?"
It's "X at p99 under Y ms," and any single number without a latency bound is
marketing.

### What to report

Per load level: offered rate (what you pushed), achieved throughput (what
actually completed), and the Lecture 24 metrics as percentiles: TTFT, TPOT
(seconds per generated token, the stream speed after the first token, from
Lecture 04), and end-to-end latency, each as p50/p90/p99, the latencies that 50,
90, and 99% of requests finished within. Plus **queue depth over time**: how
many requests are waiting at each instant, sampled continuously.

Queue depth is the diagnostic. Growing monotonically means you're past the knee,
and it tells you *before* the latency numbers do: a queue can be climbing while
percentiles are still averaging the damage away.

### Percentiles need samples

Lecture 04's caveat, now load-bearing. A p99 from 100 requests interpolates
between your two slowest samples, nearly meaningless. **Thousands of requests**
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
6. Compare your driver against vLLM's `benchmarks/benchmark_serving.py`, a
   sanity check that you're measuring the same thing.

**Predict first:** what request rate do you think is your knee? Write it down.
Most people guess high.

---

## What you should see

**A clear knee**, often at lower load than you expected.

**p99 rising much faster than p50** as you approach it. The tail always goes
first, which is why Lecture 04 insisted on percentiles.

**Queue depth growing without bound** past the knee. This is the unambiguous
saturation signal.

**Different knees for different workloads.** `long_prefill` saturates earlier than
`mixed_length`. Your capacity is a property of *traffic shape*, not just hardware.

---

## Go deeper

- **Kiely §4.5–4.5.2** (p.112–114), benchmarking tooling and practical tips.
- **Kiely §1.4.2** (p.37), end-to-end metrics, including what the client sees.
- **vLLM `benchmarks/benchmark_serving.py`**: the standard harness. Note how much
  is realistic load generation rather than timing.
- **[Open Versus Closed: A Cautionary Tale](https://www.usenix.org/legacy/event/nsdi06/tech/schroeder.html)**
  (Schroeder et al., NSDI '06), why the distinction above changes your
  conclusions. Not LLM-specific and directly applicable.
- **[Systems Performance](https://www.brendangregg.com/systems-performance-2nd-edition-book.html)**
  (Gregg): the USE method and latency analysis generally.

---

## Check yourself

1. Why does an open-loop test find problems a closed-loop test can't?
2. Past the knee, throughput is flat but latency grows. Where is the time going?
3. Why is "we handle 500 req/s" meaningless without a latency bound?
4. Why does `long_prefill` saturate earlier than `mixed_length`? *(Lecture 11.)*
5. Your p99 came from 200 requests. What's wrong with quoting it?

---

## Next

**[25b. Deployment environments](25b-deployment-environments.md)**: you
benchmarked it; now deploy it without surprise.

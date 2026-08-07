"""Phase 4 -- production. Where inference stops being kernels and becomes systems.

M4.1 api.py           OpenAI-compatible /v1/chat/completions + SSE streaming;
                      API server decoupled from the engine loop
M4.2 (bench/)         load testing: Poisson arrivals, concurrency sweep,
                      latency-vs-throughput curve, find the knee
M4.3 (bench/)         head-to-head vs. real vLLM and SGLang. Expect to lose,
                      then explain the gap with Phase 2's profiler.
M4.4 router.py        cache-aware routing: send requests to the replica likely
                      to hold the prefix. Compare hit rate vs. round-robin.
M4.5 disaggregated.py separate prefill/decode workers + KV transfer
M4.6 autoscale.py     concurrency-based scaling, cold starts, cost/M tokens
"""

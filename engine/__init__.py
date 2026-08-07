"""The engine, grown milestone by milestone through Phase 1.

M0.1 model.py, generate.py      naive loop, no cache
M1.1 cache.py                   KV cache
M1.2 sampling.py                temperature / top-k / top-p
M1.3 generate.py                static batching
M1.4 scheduler.py               continuous batching
M1.5 block_manager.py, cache.py paged KV
M1.6 block_manager.py           prefix caching + refcount/eviction
M1.7 scheduler.py               chunked prefill
M1.8 speculative.py             n-gram speculation
M1.9 model_runner (in model.py) CUDA graphs
"""

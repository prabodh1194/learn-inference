"""Phase 3 -- JAX.

JAX earns its place for one reason: it makes the compiler and the sharding
EXPLICIT and DECLARATIVE, where PyTorch/vLLM make them imperative and
hand-rolled. Learn TP as "annotate the layout, let XLA insert the collectives"
first, and vLLM's manual NCCL code becomes legible afterward.

M3.1 model.py     Qwen3 forward, pure functions + explicit params
M3.2 decode.py    lax.scan decode loop, KV cache as carry; inspect the HLO
M3.3 sharding.py  NamedSharding over a device mesh; find where scaling bends
M3.4 (-> engine/) port TP back to PyTorch by hand with NCCL collectives
M3.5 moe.py       expert parallelism: throughput-oriented vs. TP's latency
"""

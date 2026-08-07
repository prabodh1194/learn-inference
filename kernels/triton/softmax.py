"""M2.2 -- fused softmax. The Triton warmup.

Goal is the mental model, not the speedup: program ids, BLOCK_SIZE, masking
for non-power-of-2 rows, tl.load/tl.store. Benchmark vs. torch.softmax.
"""

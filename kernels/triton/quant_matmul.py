"""M2.5 -- INT8/FP8 quantized matmul.

Measure all THREE axes, not two: memory, throughput, and quality (perplexity
or a small eval). A quantization win that silently degrades output is a
regression you cannot see from a throughput chart. Book §5.1.3.
"""

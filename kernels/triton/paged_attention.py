"""M2.4 -- attention over a block table instead of contiguous memory.

The hard one, and the reason Phase 1 came first: you already know exactly what
the block table means because you built it in M1.5. Gather K/V blocks via the
table, then run the same tiled online-softmax as flash_attention.py.
"""

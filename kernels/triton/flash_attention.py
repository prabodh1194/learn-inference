"""M2.3 -- FlashAttention-style tiled attention.

Never materialize the N x N score matrix. Tile over K/V, keep a running max
and running sum (online softmax), rescale as you go.

This is the payoff of M0.4: after it works, recompute arithmetic intensity and
re-plot the roofline. You should be able to explain your own measured number.
Book §2.5.
"""

"""M2.6 (Lecture 20) -- raw CUDA."""

from __future__ import annotations


def load_extension():
    """M2.6. Compile and load the CUDA kernels via cpp_extension.load.

    Expose: softmax, reduce_naive, reduce_sequential, reduce_shuffle. The
    three reductions must agree numerically -- they differ only in memory
    access pattern, which is the whole lesson.
    """
    raise NotImplementedError("M2.6")

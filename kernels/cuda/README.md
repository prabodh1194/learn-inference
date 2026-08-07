# M2.6 — raw CUDA

Rewrite one kernel (softmax, or paged attention if you're feeling it) in CUDA C:
shared memory, warp primitives, occupancy tuning.

You will probably not beat Triton. **That is the expected result and it is fine.**
The point is knowing precisely what Triton was doing on your behalf — after this,
its abstractions stop being magic.

Build with `nvcc`, bind via `torch.utils.cpp_extension.load`.

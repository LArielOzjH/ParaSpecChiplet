# Triton Reduced-Width Kernel Attempt

## Scope

This was a proof-of-concept custom Triton implementation for the 50%-width
Qwen3 gated MLP. It computes gate and up projections in one tiled kernel and
then computes the SiLU-gated down projection in a second tiled kernel. The
experiment is deliberately separate from the official acceptance path.

## Result on RTX 4090

| Active rows | Eager reduced MLP (ms) | Triton prototype (ms) | Max absolute error |
|---:|---:|---:|---:|
| 1024 | 537 | 1,570 | 128 |
| 576 | 342 | 1,007 | 128 |
| 256 | 175 | 467 | 128 |

The refreshed batch-1/8/64 run fails the kernel gate: it is about 2.7--5.0x
slower than the eager reduced-width path and does not meet the registered
strict numerical-equivalence criterion. The initial version also exceeded the
GPU shared-memory limit;
smaller tiles and fewer pipeline stages allowed it to launch, but did not
make it competitive.

## Research implication

This negative result closes the current speedup claim. Reduced width remains promising at the
matrix-operation level, but a viable architecture needs a tensor-core-aware
kernel with explicit bf16/fp32 accumulation policy, shape-specialized layouts,
and persistent scheduling. A hand-written generic Triton decomposition is not
enough evidence for a fused-kernel speedup and must not be used as one.

The current paper should therefore report the acceptance-calibrated dataflow
and cost frontier, while treating a persistent tensor-core kernel as future
work. See [`docs/fused_kernel_gate_results.md`](fused_kernel_gate_results.md)
for the refreshed measurements.

Raw measurement: `data/gpu_triton_mlp_rtx4090.json`.

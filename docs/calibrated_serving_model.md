# Calibrated Serving-Level Schedule Model

The repository now composes measured RTX 4090 width/row latency into a fixed
batch schedule model using `paraspec.calibrated_latency.py`. For batch 64 and
16 active rows/request, the composed MLP-only estimates are:

| Schedule | Calibrated MLP latency | Relative to uniform |
|---|---:|---:|
| Uniform | 5,182 μs | 1.00x |
| Layer 2 at 50% width | 4,671 μs | 0.901x |
| Layers 2+3 at 50% width | 4,160 μs | 0.803x |

The two-layer schedule is not acceptance-safe on the current workload
(`S1=0.5625`), so its lower modeled latency cannot be claimed as a useful
configuration. The single layer-2 schedule passes the current `S1` gate and
is the only calibrated frontier point with both acceptance and compute-side
support.

This model is intentionally transparent: it composes fixed-shape single-layer
MLP measurements and leaves dense attention as a separate additive term. It
does not extrapolate across batch sizes, model workloads, or queueing states,
and it is not an end-to-end serving speedup. Its architectural purpose is to
make the scheduler's decision auditable and to expose where a fused kernel or
dense fallback is required.

The implementation is in `paraspec/calibrated_latency.py` and
`scripts/analyze_calibrated_schedule_latency.py`.

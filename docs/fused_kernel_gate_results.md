# Fused Kernel Gate: Final Decision

Date: 2026-08-14

## Experiment

The existing two-stage Triton gated-MLP prototype was rerun on the RTX 4090
for the Qwen3-4B-DFlash-b16 MLP at 50% intermediate width. It was compared
with the eager PyTorch reduced-width reference at the same active-row count.

## Results

| Batch | Active rows/request | Reference (ms) | Triton (ms) | Triton/reference | Max abs. error |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 69.14 | 337.71 | 4.88× | 32 |
| 1 | 9 | 67.94 | 334.81 | 4.93× | 16 |
| 1 | 4 | 66.81 | 333.78 | 5.00× | 4 |
| 8 | 128 | 110.55 | 360.03 | 3.26× | 64 |
| 8 | 72 | 91.55 | 341.25 | 3.73× | 64 |
| 8 | 32 | 67.91 | 338.28 | 4.98× | 64 |
| 64 | 1024 | 514.46 | 1485.05 | 2.89× | 128 |
| 64 | 576 | 330.54 | 957.48 | 2.90× | 128 |
| 64 | 256 | 167.90 | 457.10 | 2.72× | 128 |

Against an fp32 accumulation reference for 1024 active rows, the maximum
absolute error was `55.94`, mean error `0.29`, and relative maximum error
approximately `0.26%`. The error reflects different accumulation and bf16
conversion paths, but the prototype has not been integrated into the official
DFlash verifier or accepted under a registered numerical-equivalence criterion.

## Decision

The prototype fails the performance gate at every tested batch. It is also not
an official-loop correctness result. Therefore:

- do not claim fused-kernel speedup;
- do not claim a correctness-passing custom Triton kernel;
- retain isolated reduced-width measurements as compute-side headroom;
- retain SAGE-DFlash as an acceptance-calibrated dataflow/cost study with
  dense fallback;
- treat a tensor-core-aware persistent kernel as future work, not as a required
  contribution for the current paper.

Raw artifacts:

- [`data/triton_gate_b1.json`](../data/triton_gate_b1.json)
- [`data/triton_gate_b8.json`](../data/triton_gate_b8.json)
- [`data/triton_gate_b64.json`](../data/triton_gate_b64.json)
- [`scripts/benchmark_triton_mlp.py`](../scripts/benchmark_triton_mlp.py)

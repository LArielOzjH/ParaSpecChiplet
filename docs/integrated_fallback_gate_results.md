# Integrated MLP Gate and Dense-Fallback Results

Date: 2026-08-14

## Implementation

`paraspec.official_selective.install_mlp_gates` now supports an optional
`dense_fallback_fraction`. For each draft layer, the controller computes the
fraction of active position rows in the selected depth schedule:

- active fraction below the threshold: evaluate the original MLP densely;
- active fraction at or above the threshold: gather active rows, run one
  grouped MLP call, and scatter the result back;
- attention remains unchanged.

The strict comparison is intentional: equality at the registered crossover
uses grouped execution. The official probe exposes this as
`--dense-fallback-fraction`.

## Acceptance probe

The official Qwen3-4B/DFlash-b16 verifier was run on the 8-prompt, 96-token
held-out workload with a fallback threshold of `0.6`:

| Schedule | Mean accepted prefix | `S1` | `S2` | `S4` |
|---|---:|---:|---:|---:|
| Uniform | 1.4219 | 0.6375 | 0.3781 | 0.0938 |
| Protected8 conservative | 1.4448 | 0.6435 | 0.3849 | 0.0946 |
| Protected8 staircase | 1.4371 | 0.6478 | 0.3805 | 0.0943 |
| Protected4 moderate | 1.4525 | 0.6551 | 0.3797 | 0.0981 |

These are acceptance results, not serving-speed results. The small differences
from the no-fallback trace come from the official loop's cycle behavior; the
fallback path does not alter target verification.

## CUDA microbenchmark

A fixed-shape, single-layer benchmark used 9 active rows per request and
compared dense MLP, eager gather/scatter grouped MLP, and CUDA-graph replay:

| Batch | Dense (ms) | Eager grouped (ms) | CUDA graph (ms) |
|---:|---:|---:|---:|
| 1 | 179.03 | 181.76 | 176.32 |
| 8 | 213.20 | 215.03 | 209.78 |
| 64 | 1038.22 | 684.95 | 680.02 |

The fixed-shape graph shows a clear compute-side benefit at batch 64, while
eager gather/scatter does not pay for launch/scatter overhead at batch 1 or 8.
This supports an occupancy gate and persistent/fixed-shape execution, but it
does not establish end-to-end DFlash throughput or a chiplet advantage.

Raw artifacts:

- [`data/heldout_mlp_gating_fallback06.jsonl`](../data/heldout_mlp_gating_fallback06.jsonl)
- [`data/fallback_mlp_b1_8_64.json`](../data/fallback_mlp_b1_8_64.json)

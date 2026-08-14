# Integrated Official-Loop Gate

Date: 2026-08-14

## Scope

The official Qwen3-4B/DFlash-b16 serving loop was run on the 8-prompt,
96-token held-out workload with the attention-preserving MLP gate and a `0.6`
active-row dense-fallback threshold. This is the closest current integration
test: target verification is unchanged, while selected MLP calls gather active
rows or fall back to dense execution.

## Acceptance

| Schedule | Mean accepted prefix | `S1` | `S2` | `S4` |
|---|---:|---:|---:|---:|
| Uniform | 1.4219 | 0.6375 | 0.3781 | 0.0938 |
| Protected8 conservative | 1.4448 | 0.6435 | 0.3849 | 0.0946 |
| Protected8 staircase | 1.4371 | 0.6478 | 0.3805 | 0.0943 |
| Protected4 moderate | 1.4525 | 0.6551 | 0.3797 | 0.0981 |

The integrated path preserves the acceptance-compatible summary of the
protected schedules. This validates the dataflow/verification interface, not
its hardware speed.

## Timing interpretation

The event trace records official-loop average time per output token. Per-prompt
means with the Python hook were noisy and schedule-dependent; for example, the
uniform and protected8 staircase runs were approximately `9129 us` and
`9436 us` respectively in one held-out run. The hook still pays Python
dispatch, tensor gather/scatter, and unfused MLP costs, so this result does not
establish a speedup. It is evidence that acceptance and the target-verifier
contract survive integration, while the fused tensor-core path remains an
open hardware gate.

The measured fixed-shape CUDA microbenchmark remains a separate datapath
result: at batch 64 and 9 active rows/request, CUDA-graph grouped MLP was
`680.02 ms` versus dense `1038.22 ms` for one layer. It must not be reported as
end-to-end DFlash serving throughput.

## Decision

The paper can claim:

- an official-loop acceptance-tested selective MLP interface;
- a dense fallback policy and its correctness behavior;
- a throughput-regime fixed-shape compute opportunity.

The paper cannot yet claim end-to-end speedup. The remaining decisive gate is a
correctness-tested fixed-shape/fused implementation measured against dense
fallback at batch 1, 8, and 64, with draft, verification, queue, and compaction
costs reported separately. If that gate is not positive, present SAGE-DFlash as
an acceptance-calibrated architecture/dataflow and cost study.

Raw trace:

- [`data/heldout_mlp_gating_fallback06.jsonl`](../data/heldout_mlp_gating_fallback06.jsonl)

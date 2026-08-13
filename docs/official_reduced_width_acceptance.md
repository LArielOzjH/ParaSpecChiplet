# Official Reduced-Width MLP Acceptance

The reduced-width MLP path was connected to the official DFlash serving loop.
Attention remains dense; selected draft MLPs use truncated gate/up/down
projection widths. The hook implementation is functional acceptance evidence,
not an end-to-end hardware timing result.

## Setup

- Target: Qwen3-4B
- Draft: Qwen3-4B-DFlash-b16
- Draft depth: 5
- Block size: 16
- Workload: 12 prompts, 96 generated tokens
- Width: 50% of the 9728 intermediate channels
- GPU: RTX 4090

| Schedule | Mean accepted prefix | `S1` | `S2` | `S4` |
|---|---:|---:|---:|---:|
| Uniform | 1.4525 | 0.6095 | 0.3368 | 0.1033 |
| Layer 2 at 50% width | 1.4412 | 0.6103 | 0.3320 | 0.1072 |
| Layers 2+3 at 50% width | 1.3145 | 0.5625 | 0.2988 | 0.0898 |

The single-layer result passes the current early-prefix gate within the
observed workload, while the joint schedule loses acceptance. This is
consistent with the all-or-nothing and output-scaling experiments: block
fidelity has interaction effects, and a static per-layer ranking is not enough.

The trace's serving-loop latency is diagnostic only. The width hook runs a
Python replacement path and does not establish speedup. The corresponding
CUDA microbenchmark shows that regular reduced-width matrix multiplications
can materially reduce MLP latency at batch 64, so the remaining experiment is
an integrated fused implementation with a jointly safe schedule table.

Raw traces:

- `data/official_qwen3_4b_mlp_width_l2_05.jsonl`
- `data/official_qwen3_4b_mlp_width_l23_05.jsonl`

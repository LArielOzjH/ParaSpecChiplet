# Fixed-Shape Specialization Measurements

Date: 2026-08-14

## Purpose

The chiplet hypothesis needs a physical specialization benefit beyond merely
partitioning the same compute. We measured fixed-shape reduced-width MLPs on
the RTX 4090 as a proxy for a dedicated narrow MLP lane.

The benchmark evaluates one draft MLP in isolation. It does not include
chiplet links, target verification, attention, request queueing, or acceptance.

## Measured latency

Selected results from Qwen3-4B-DFlash-b16 are below. `Dense` is the full-width
MLP on the same active rows; `Reduced` uses truncated gate/up/down projections.

| Batch | Active rows/request | Width | Dense (ms) | Reduced (ms) | Ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 50% | 178.43 | 65.25 | 0.366× |
| 1 | 16 | 25% | 178.38 | 48.53 | 0.272× |
| 8 | 16 | 50% | 213.84 | 106.06 | 0.496× |
| 8 | 16 | 25% | 213.97 | 55.94 | 0.261× |
| 64 | 16 | 50% | 1037.59 | 523.32 | 0.504× |
| 64 | 16 | 25% | 1036.92 | 271.41 | 0.262× |
| 64 | 8 | 50% | 539.28 | 285.54 | 0.529× |
| 64 | 8 | 25% | 538.01 | 167.60 | 0.312× |

The data demonstrates a real compute-side specialization opportunity. It does
not demonstrate that the corresponding fidelity schedule is safe: the held-out
official acceptance probe found that scaling block 2, block 3, or both to 50%
was below uniform. A dedicated lane therefore needs a jointly validated
schedule descriptor and dense fallback.

## Chiplet decision

This experiment does not promote chiplets by itself. The strongest defensible
interpretation is:

1. reduced-width lanes are worth modeling as a datapath option;
2. block/position fidelity must be validated independently using target
   verification;
3. chiplets are justified only if the specialized lane's utilization or
   sharing gain exceeds the link/synchronization break-even boundary;
4. the grouped monolithic fabric remains the equal-resource baseline.

Raw artifacts:

- [`data/width_specialization_b1.json`](../data/width_specialization_b1.json)
- [`data/width_specialization_b8.json`](../data/width_specialization_b8.json)
- [`data/width_specialization_b64.json`](../data/width_specialization_b64.json)
- [`scripts/benchmark_mlp_width_rows.py`](../scripts/benchmark_mlp_width_rows.py)

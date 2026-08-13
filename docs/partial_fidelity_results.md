# Partial MLP Fidelity Results

The all-or-nothing MLP bypass experiments showed that safe single-block
choices do not compose independently. We therefore tested a softer primitive:
the selected MLP output is multiplied by a fidelity factor `alpha` while
attention remains dense.

## Official setup

- Target: Qwen3-4B
- Draft: Qwen3-4B-DFlash-b16
- Block size: 16
- Draft depth: 5
- Workload: same 12 prompts and 96 generated tokens as the block ablation
- Verifier: official DFlash serving loop on RTX 4090

The hook still executes the complete MLP before scaling its output. These are
acceptance probes, not measured compute savings or speedups.

## Results

| Schedule | Mean accepted prefix | `S1` | `S2` | `S4` |
|---|---:|---:|---:|---:|
| uniform | 1.4525 | 0.6095 | 0.3368 | 0.1033 |
| layer 2, `alpha=0.5` | 1.4657 | 0.6216 | 0.3326 | 0.0998 |
| layer 2, `alpha=0.25` | 1.4792 | 0.6375 | 0.3521 | 0.0958 |
| layer 3, `alpha=0.5` | 1.4454 | 0.6082 | 0.3443 | 0.0969 |
| layers 2+3, `alpha=0.5` | 1.5127 | 0.6335 | 0.3602 | 0.1102 |
| layers 2+3, `alpha=0.25` | 1.3241 | 0.6028 | 0.3281 | 0.0909 |

The results indicate that partial fidelity can preserve early-prefix
acceptance better than zeroing the same updates. The joint `alpha=0.5`
schedule is especially encouraging, while the joint `alpha=0.25` schedule
shows that fidelity changes still interact nonlinearly. The apparent gains
are not claimed as performance improvements because the current implementation
does not reduce MLP work.

## Architecture implication

The candidate mechanism should expose a small set of precompiled fidelity
levels per draft block (for example full, half, and quarter), then select a
joint schedule subject to a measured prefix-survival constraint. Hardware
realizations include lower-precision MLP lanes, reduced-width matrix tiles,
or partial lane participation. A chiplet implementation is useful only if
these fidelity levels can be served with less work and acceptable activation
traffic.

The next decisive experiment is a fused low-precision or reduced-lane MLP
kernel with the same schedule table. Until then, these results support the
acceptance side of the proposal only.

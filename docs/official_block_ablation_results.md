# Official Draft-Block Ablation Results

## Setup

- Target: Qwen3-4B
- Draft: Qwen3-4B-DFlash-b16
- Draft depth: 5 Transformer blocks
- Block size: 16
- GPU: RTX 4090
- Workload: 12 prompts, 96 generated tokens per experiment
- Verifier: official `z-lab/dflash` serving loop

The raw trace is `data/official_qwen3_4b_block_ablation.jsonl`. Each experiment
uses the same target verifier. A forward hook replaces the selected draft
layer output with its input; the selected layer is still executed. Therefore
the latency fields are diagnostic only and are not a hardware speedup result.

## Results

| Experiment | Cycles | Mean accepted prefix | `S1` | `S2` | `S4` | Diagnostic latency (us) |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 484 | 1.4525 | 0.6095 | 0.3368 | 0.1033 | 9,356 |
| bypass layer 0 | 1,134 | 0.0176 | 0.0176 | 0.0000 | 0.0000 | 21,077 |
| bypass layer 1 | 624 | 0.8830 | 0.4551 | 0.1875 | 0.0593 | 12,015 |
| bypass layer 2 | 584 | 0.9846 | 0.5719 | 0.2500 | 0.0291 | 11,278 |
| bypass layer 3 | 585 | 0.9897 | 0.5436 | 0.2547 | 0.0393 | 10,965 |
| bypass layer 4 | 613 | 0.8940 | 0.5334 | 0.2072 | 0.0359 | 11,602 |

## Interpretation

The evidence supports a non-uniform block value hierarchy:

1. The first draft block is highly sensitive. Bypassing it reduces `S1` from
   `0.6095` to `0.0176`, so a design that skips early blocks is not viable.
2. Later blocks are less catastrophic than block 0, but none is free: every
   full-block bypass lowers mean accepted prefix and early-prefix survival.
3. The result is not evidence for full-layer skipping speedup. The Python hook
   still pays for the bypassed layer, and the increased cycle count is expected
   when acceptance falls.
4. The useful architecture question is now **how much fidelity each block
   needs**, not whether an entire block can be removed. Candidate mechanisms
   are MLP-only approximation, block-specific precision, and shared versus
   specialized execution lanes.

## Next gate

Run the same official loop with only one block's MLP update bypassed at a time,
then with block-specific reduced precision. Keep attention dense until the
acceptance gate passes. A candidate is viable only if it reduces measured
draft work in a fused implementation while preserving the early-prefix
survival tolerance.

The current result is therefore a positive motivation result for block-aware
heterogeneity, but not yet a performance claim or a justification for
full-layer chiplet skipping.

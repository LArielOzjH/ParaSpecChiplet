# DAC'27 Evidence Matrix and Idea Hierarchy

Date: 2026-08-14

This document is the current source of truth for the paper direction. It
separates motivation evidence, architecture evidence, and claims that remain
unproven.

## Recommended paper

**SAGE-DFlash: Prefix-Survival-Aware MLP Scheduling for Block-Parallel
Speculative Decoding**

### One-sentence thesis

DFlash executes every draft position uniformly even though target verification
commits a prefix; a jointly validated position/depth MLP schedule can preserve
the valuable prefix, while a grouped fabric and occupancy gate convert only
profitable schedules into hardware work.

### Motivation

Official ablations show that draft blocks do not have equal marginal value:
bypassing block 0 is catastrophic (`S1=0.0176`), while bypassing only block 2's
MLP preserved the early-prefix metric on the original workload (`S1=0.6096`
versus `0.6095` uniform). This is motivation, not a static ranking rule:
held-out 50%-MLP scaling of block 2, block 3, or both was below uniform.

### Challenge

Bidirectional attention couples positions inside a draft block, so tail
positions cannot simply be deleted. Fidelity decisions also interact across
blocks, and gather/scatter overhead makes selective execution unprofitable at
low occupancy.

### Architecture

1. Keep bidirectional attention dense.
2. Carry a finite, jointly validated position/depth schedule descriptor.
3. Execute MLP rows through grouped full/reduced-depth lanes.
4. Estimate occupancy and use dense fallback below the measured crossover.
5. Compare against an equal-resource grouped monolithic fabric.

Chiplets are an optional physical realization only if they provide a measured
specialization/utilization or area/energy benefit above the traffic break-even
boundary.

## Evidence table

| Question | Current evidence | Interpretation | Status |
|---|---|---|---|
| Are blocks/positions equally valuable? | Full-block and MLP-only ablations show large, non-uniform acceptance losses. | Strong motivation for heterogeneous fidelity. | Supported |
| Is a fixed block-width ranking safe? | Held-out block 2 ×0.5: mean `1.3720`, `S1=0.6250`; uniform mean `1.4219`, `S1=0.6375`. | Static block specialization is not robust. | Rejected |
| Is a position/depth schedule acceptance-compatible? | Protected8 staircase: no-fallback mean `1.4448`, `S1=0.6530`; fallback probe mean `1.4371`, `S1=0.6478`; uniform `1.4219`, `0.6375`. | Strongest current fidelity candidate. | Supported as acceptance evidence |
| Does selective execution have compute headroom? | Batch-64 CUDA graph grouped MLP: `680.02 ms` vs dense `1038.22 ms` for 9 active rows/request. | Fixed-shape grouped path can help in throughput regime. | Supported as microbenchmark |
| Does grouping always help? | Batch 1/8 row grouping does not pay back; row sweep crosses near 12/16 active rows at batch 64. | Occupancy gate and dense fallback are required. | Supported |
| Does simple state adaptation help? | Previous-prefix threshold positive in a short probe, negative on 96-token held-out follow-up. | Do not claim adaptive policy gain. | Rejected for now |
| Do chiplets automatically help? | Equal-resource break-even requires `1.005–1.021×` effective gain after overhead. | Chiplets need an independent specialization/utilization result. | Conditional |
| Is end-to-end speedup established? | Official-loop gate preserves acceptance, but Python gather/scatter timing is not a hardware result; the Triton prototype is 2.7--5.0× slower than eager reduced MLP. | Report calibrated proxy/microbenchmark only and remove speedup claim. | Rejected for current paper |

## Idea hierarchy

### Primary: survival-aware grouped monolithic fabric

Motivation: uniform draft work conflicts with prefix commitment.

Contribution: a schedule table jointly filtered by prefix survival, a dense
attention/heterogeneous MLP dataflow, and an occupancy-aware grouped/dense
execution policy.

Required proof: correctness-tested fused or fixed-shape path beats dense
fallback in the stated throughput batch regime, with acceptance and queue
costs included.

### Supporting mechanism: specialized MLP lanes

Fixed-width measurements show substantial isolated MLP headroom, so full and
reduced-fidelity lanes are a meaningful implementation point. This mechanism
does not independently establish a safe schedule or a chiplet advantage.

### Conditional extension: chiplet fabric

Promote only if an equal-resource, traffic-aware evaluation demonstrates a gain
above the break-even boundary through parallel schedule-class service, lane
specialization, or area/energy sharing. Otherwise retain the monolithic fabric
and explicitly kill the chiplet headline.

### Killed or demoted directions

- grammar-aware decoding: not needed for the architecture opportunity;
- sparse-head execution: insufficiently aligned with DFlash's dense coupling;
- static block-width specialization: failed held-out acceptance;
- simple previous-prefix adaptive selector: failed long held-out trace;
- chiplet-first framing: no independent benefit yet.

## Claims discipline

The paper may claim unequal block value, acceptance-calibrated candidate
schedules, dense-attention preservation, occupancy-dependent grouped
execution, and measured compute-side headroom. It may not claim end-to-end
throughput, universal schedule safety, generalized adaptive-policy gain,
chiplet advantage, or fused-kernel speedup in the current submission.

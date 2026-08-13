# DAC'27 Submission Blueprint

## Working title

**SAGE-DFlash: Prefix-Survival-Aware MLP Scheduling for Block-Parallel
Speculative Decoding**

The title deliberately names the execution/dataflow problem. Chiplets are an
optional implementation point, not the premise of the paper.

## Core claim

DFlash creates a mismatch between computation and commitment: every block
position is computed in parallel, but only a contiguous accepted prefix is
valuable. A useful accelerator should therefore spend draft-block MLP work
according to measured prefix value while preserving dense bidirectional
attention. The proposed execution substrate is:

1. shared lower-layer/block-wide execution;
2. dense attention for every block position;
3. jointly validated position/depth MLP schedules;
4. cross-request grouping of equal fidelity schedules;
5. an occupancy-gated dense fallback when compaction is not profitable;
6. a finite schedule table that can be driven by a static baseline, an
   offline oracle, or a state/workload-conditioned selector.

The primary baseline is an equal-resource monolithic accelerator. A chiplet
fabric is evaluated only as an optional way to realize the shared backbone and
heterogeneous MLP lanes.

## Motivation backed by current evidence

The official Qwen3-4B/DFlash-b16 CUDA trace contains 480 decode cycles over 12
prompts. The accepted-prefix survival curve is `S1=0.596`, `S2=0.346`, and
`S4=0.096`; per-prompt mean accepted prefixes range from 0.865 to 2.345. Thus
uniform per-position execution is not a faithful model of committed value.

On the same official serving loop, protected8 staircase MLP gating reduced
nominal MLP rows by 25% while changing mean accepted prefix from 1.259 to
1.261 in the measured workload. This is acceptance-compatible dataflow
evidence, not an end-to-end speedup claim.

The hardware boundary is equally important. On an RTX 4090, batch 1/8
gather/scatter execution was slower than dense for every tested row count. At
batch 64, grouped execution became faster at 12/16 active rows and below. The
current evidence therefore defines a throughput-serving regime and motivates
cross-request grouping plus dense fallback.

## Architecture

```text
target hidden features
          |
   shared DFlash backbone / dense attention
          |
   survival-map scheduler
       /             \
  protected lanes    tail grouped lanes
       \             /
        batch-aware MLP compactor
                 |
          target verifier
```

The scheduler operates on a joint block-fidelity vector, not on individual
token pruning. It must expose three decisions to hardware:

- MLP width/fidelity for the current draft layer;
- whether the active-row group is large enough for compaction;
- whether requests should wait briefly for schedule coalescing or use dense
  execution immediately.

The occupancy gate is a first-class architectural decision. The current
normalized sweep crosses from grouped loss to grouped benefit near 40--50%
heterogeneous-schedule occupancy; the controller must be able to select dense
execution below that crossover.

Attention remains dense because tail hidden states can provide context to
protected prefix positions. Only the position-wise residual MLP update is
selectively executed in the primary design.

## Contributions to claim if the remaining gate passes

### 1. Acceptance-calibrated DFlash fidelity model

Define `S_i = P(A >= i)` and evaluate joint block-fidelity actions by committed
value per cost. Reject schedules below a registered prefix-acceptance
threshold; do not assume independent per-block scores compose.

### 2. Grouped heterogeneous MLP dataflow

Provide a concrete execution path that preserves dense attention while
executing reduced-width/partial-fidelity MLPs and grouping compatible
requests. Include a measured dense fallback and schedule-group queue, so the
design remains meaningful outside an ideal sparse-MAC model.

### 3. Equal-resource architecture study

Compare dense monolithic, grouped monolithic, and optional chiplet mappings
with compute, activation movement, synchronization, queue fill, and router
costs. The chiplet result is a conditional finding, not a required positive
claim.

## Required evaluation matrix

| Axis | Required cases | Primary metric |
|---|---|---|
| Acceptance | uniform, protected8 conservative, protected8 staircase, protected4 | `S1`, `S2`, `S4`, mean accepted prefix |
| Execution | dense, eager grouped, fused/persistent grouped | draft latency, MLP work |
| Batch | 1, 8, 16, 32, 64 | committed tokens/s, queue fill |
| Mixture | homogeneous, 25/50/75% staircase, trace-derived | active-row utilization, tail latency |
| Policy | dense fallback, static schedule, state-aware oracle | value/cost, controller overhead |
| Fabric | monolithic, grouped monolithic, chiplet | cycles, bytes, sync, energy/area proxy |
| Workload | systems, coding, math/reasoning, long context | cross-workload stability |

The fused/persistent reduced-width implementation is the decisive missing
artifact. Current evidence already includes a reduced-width CUDA calibration,
but not a correctness-passing fused kernel or end-to-end speedup.
Until it exists, report the current row-latency measurements as hardware
calibration and not as accelerator speedup.

A fixed-shape CUDA Graph probe is an intermediate result: graph-safe
`index_copy_` replay removes some launch overhead, but it still has no saving
at batch 1/8. It should be treated as an implementation baseline for the
custom persistent lane, not as the final fused design.

## Baselines and novelty boundary

Required baselines:

- vanilla uniform-depth DFlash;
- uniformly shallow DFlash;
- fixed protected-prefix staircase;
- dynamic draft-length or verification-pruning baseline;
- ideal zero-overhead grouped execution;
- equal-resource monolithic dense accelerator;
- explicit chiplet mapping with link and synchronization cost.

Do not claim novelty for positional loss weighting, generic early exit,
dynamic draft length, generic token pruning, quantization, low-rank projection,
or chiplets alone. The differentiator is physical execution under
prefix-survival and bidirectional-context constraints.

## Decision tree

1. If no jointly validated fidelity schedule preserves early-prefix acceptance
   beyond the registered tolerance, kill heterogeneous block fidelity and
   retain only dense-attention profiling.
2. If fused grouped execution does not beat dense in the target throughput
   regime, kill the grouped hardware claim and retain the acceptance result as
   motivation only.
3. If state-conditioned schedules do not beat static schedules after controller
   cost, remove adaptive scheduling from the main paper.
4. If chiplet traffic and synchronization lose to equal-resource monolithic,
   remove chiplets from the headline and publish the grouped monolithic design.
5. If all gates pass, present SAGE-DFlash as an acceptance-calibrated grouped
   fidelity accelerator; chiplets remain an optional physical organization.

## Current status

Supported now:

- official serving-loop acceptance trace;
- official block ablation and reduced-width MLP acceptance probes;
- batch/schedule GPU microbenchmarks;
- trace-driven queue sensitivity model;
- calibrated row policy and dense fallback;
- equal-resource chiplet/monolithic analytical cost model.

Not supported now:

- fused/persistent reduced-width kernel;
- low-batch speedup;
- end-to-end throughput improvement;
- causal proof that state-aware scheduling improves acceptance;
- chiplet advantage after measured traffic and synchronization.

The paper should not be written as complete until the missing artifacts are
measured or the corresponding claims are explicitly removed. If the fused
kernel gate fails, the defensible submission is an acceptance-calibrated
architecture/cost study with monolithic grouped execution as the proposed
organization and chiplets explicitly demoted.

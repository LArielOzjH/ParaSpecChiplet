# DAC'27 Idea Catalog

## Recommended: SAGE-DFlash

**One-line idea:** select a jointly safe vector of draft-block MLP fidelity
levels using official prefix acceptance and measured execution cost, then
serve compatible requests in grouped monolithic lanes with dense fallback.

**Motivation.** DFlash parallelizes a block, but verification commits a prefix.
The draft stack also spends uniform MLP work on blocks with unequal marginal
value. The mismatch is architectural: value is prefix-conditioned while work
is uniform.

**Challenge.** Bidirectional attention couples all positions, and individually
safe block approximations do not compose independently. The scheduler must
choose joint schedules and pay for queueing, movement, and synchronization.

**Current evidence.** On the original official Qwen3-4B/DFlash-b16
calibration, layer 2 at 50% MLP width gives `S1=0.6103` versus `0.6095`
uniform; layers 2+3 at 50% gives `S1=0.5625`. A new 8-prompt held-out screen
is cautionary: layer 2 gives mean prefix `0.9624` versus `1.0234` uniform,
layer 3 gives `0.8696`, and layers 2+3 gives `0.8561`. The evidence supports
unequal and state/workload-dependent block value, not a fixed safe layer. The
calibrated batch-64 MLP proxy is 9.9% lower for the single-layer schedule,
before dense attention and service overhead.

**Architecture.** Dense attention lanes, full/half-width MLP lanes, schedule
table, cross-request coalescing, and dense fallback. The minimum paper claim
does not require chiplets or an adaptive policy.

**Kill condition.** No state/workload-conditioned schedule survives the
acceptance threshold, or a correctness-tested grouped implementation loses to
dense fallback/monolithic execution after all overheads. A static layer
ranking is already considered insufficient.

## Conservative fallback: dense-attention fidelity table

Use a static finite table of safe width vectors, selected offline from held-out
official traces. This is the fallback if state conditioning is unstable or
the controller is not worth its overhead. It still has a hardware/dataflow
story because it requires regular reduced-width MLP lanes and schedule-aware
batching.

## Optional extension: elastic heterogeneous fabric

Map full/half-width MLP lanes to separate physical regions or chiplets only if
the mapping provides parallel specialization, area/energy sharing, or workload
isolation. The current equal-resource sweep shows chiplet cycles above
monolithic cycles at all tested link bandwidths, so chiplets are not a positive
claim today.

## Optional extension: verifier-feedback scheduling

Use recent accepted-prefix history to choose among prevalidated schedules. It
is not the novelty anchor: current state-conditioned evidence is descriptive,
and existing work already studies dynamic speculation budgets. Include only if
held-out phase adaptation beats the static table after controller cost.

## Explicitly rejected directions

- grammar-aware decoding;
- sparse-head execution;
- arbitrary full-layer skipping;
- independent per-layer ranking without joint validation;
- generic `torch.compile` as a fused-kernel argument;
- the current naive Triton kernel as a speedup result;
- chiplet-first framing.

## Submission claim boundary

The strongest current paper can claim an acceptance-calibrated architecture
hypothesis, official block-fidelity evidence, measured GPU compute-side
calibration, and a traffic-aware monolithic/chiplet comparison. It cannot yet
claim end-to-end throughput, low-batch benefit, chiplet advantage, or a
correctness-passing fused kernel.

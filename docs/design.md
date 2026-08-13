# Survival-Aware Heterogeneous DFlash

## Problem

In block-parallel speculative decoding, a draft block of `B` positions is computed in one forward pass and verified as a contiguous prefix. If position `i` is wrong, positions after `i` cannot be committed in that cycle. This creates a value asymmetry that is distinct from raw token accuracy: the relevant quantity is prefix survival, not independent per-position correctness.

For a verification trace with accepted prefix length `A`, define:

```text
S_i = P(A >= i)
h_i = P(A = i - 1 | A >= i - 1)
```

`S_i` is the probability that position `i` can contribute a committed token. A layer or hardware action should be evaluated by its change in `S_i` per added cost, not by equal treatment of all positions.

## Architecture thesis

Use a two-dimensional survival map over draft layer and block position. The lower draft layers remain shared across all positions to preserve bidirectional context. Upper layers are scheduled asymmetrically:

- high-survival prefix positions receive full-depth/full-fidelity execution;
- low-survival tail positions may receive fewer upper layers or reduced precision;
- a confidence controller can promote a tail position when its predicted survival is high;
- target verification remains authoritative, so the output distribution is unchanged by construction; only draft quality and acceptance may change.

The chiplet substrate is a means to make this heterogeneity physically efficient:

- shared backbone chiplet for dense lower-layer work;
- full-fidelity prefix chiplet;
- low-cost tail/refinement chiplet;
- scheduler/router chiplet holding the survival map and routing metadata;
- target verifier remains a separate consumer.

The design must be compared against a monolithic accelerator with the same total compute and memory budget. Chiplets are not a contribution by themselves.

## Runtime controller boundary

The controller is intentionally a replay oracle, not an unverified learned
policy. Each candidate schedule supplies held-out prefix-survival estimates and
a traffic-aware cost. Recent acceptance history provides the protected-prefix
constraint; candidates that drop the protected prefix beyond the configured
tolerance are rejected, and the remaining candidates are ranked by predicted
committed value per cost. This separates the architecture question (how to
route and execute a selected schedule) from the prediction question (whether
survival can be estimated cheaply enough online).

The replay entry point is:

```bash
python scripts/replay_schedule_controller.py \
  --trace data/official_qwen3_4b_dflash_trace.jsonl \
  --options options.json --protected-prefix 4 --max-prefix-drop 0.02
```

The controller must be compared against a static schedule and an oracle with
zero controller overhead. If its selected schedule does not beat static
staircase execution after router and synchronization cost, the adaptive
controller is not a paper contribution.

## Schedule frontier oracle

Before running every selective-depth candidate on a real model, the repository
can enumerate monotone depth vectors and rank them using an additive
layer-position ablation proxy:

```bash
python scripts/enumerate_schedule_frontier.py \
  --trace data/layer_position_trace.json \
  --output data/schedule_frontier.json \
  --min-depth 1 --protected-prefix 4 \
  --macs-per-layer 100 --compute-macs-per-cycle 100 \
  --activation-bytes-per-position 4096 --link-bytes-per-cycle 512 \
  --synchronization-cycles 20
```

The output is a candidate frontier, not an accuracy result. Its additive
survival prediction is useful for narrowing the 5-by-16 official DFlash design
space; each frontier point must later be remeasured with actual selective-depth
execution and target verification.

## Block-state conditioning

Block importance is not assumed to be a function of block index. The entering
state of a block is observable from the previous verification cycle, so traces
can be partitioned by previous accepted-prefix length (and later by confidence
or context buckets):

```bash
python scripts/analyze_state_conditioned_trace.py \
  --trace data/official_qwen3_4b_dflash_trace.jsonl \
  --output data/state_conditioned_survival.json \
  --bucket-size 2
```

This reports `P(A >= i | entering-state)`, expected committed-prefix value, and
state-to-outcome counts. The result is descriptive rather than causal: a
state-conditioned difference justifies an adaptive scheduling experiment, but
does not prove that changing the schedule causes the difference. The adaptive
controller should use held-out state-conditioned traces and report phase
recovery, calibration error, and controller overhead.

For state-specific replay, provide an options JSON keyed by the emitted state
labels and run:

```bash
python scripts/replay_state_controller.py \
  --trace data/official_qwen3_4b_dflash_trace.jsonl \
  --options state_options.json --output state_decisions.json \
  --bucket-size 2 --protected-prefix 4 --max-prefix-drop 0.02
```

The options for each state must contain separately measured survival estimates;
reusing one global estimate would erase the very block-state effect being
tested.

## Monolithic baseline and chiplet crossover

The repository now includes an equal-resource monolithic baseline that models
idle lanes when positions use different depths. Sweep link bandwidth,
activation multicast reuse, and router overhead before making any chiplet
claim:

```bash
python scripts/sweep_fabric_tradeoff.py \
  --depths 3 3 2 2 2 1 1 1 --output data/fabric_tradeoff.json \
  --link-bandwidths 32 128 512 --reuse 1 2 4 \
  --macs-per-layer 100 --compute-macs-per-cycle 100 \
  --activation-bytes-per-position 4096 --synchronization-cycles 20 \
  --shared-lower-depth 1 --router-cycles-per-position 0.25
```

This analytical crossover is a screening tool. A chiplet contribution requires
calibration with measured activation bytes, actual synchronization points, and
an equal-resource monolithic implementation or simulator.

## Preferred microarchitecture: dense attention, gated MLP

The activation probe suggests a safer hardware primitive than skipping an
entire DFlash layer. Keep the block-wide bidirectional attention dense so tail
positions continue to provide context, but gate the residual MLP update by the
survival schedule. The cost oracle `estimate_mlp_gated_cost` models this as
dense attention at the maximum depth plus MLP work only for executed
position/layer pairs. This is the primary implementation candidate for the
next GPU/kernel study; whole-layer skipping remains a more aggressive
ablation, not the default architecture.

The checked-in illustrative sweep (`data/fabric_tradeoff_example.json`) is
intentionally not favorable to chiplets: with 4096 bytes per position,
20 synchronization cycles, and router cost 0.25 cycles/position, the
equal-resource monolithic baseline is 44 cycles. The best tested chiplet point
(512 B/cycle link, reuse 4) is 46 cycles; lower bandwidth or reuse is much
worse. This is an early kill signal for a naive split fabric and motivates
activation multicast/reuse as a necessary mechanism rather than a cosmetic
feature.

## Cost-model boundary

The chiplet oracle models four terms separately: draft compute, activation-link
cycles, synchronization, and router/scheduler cycles. It can also account for
lower-layer execution shared once across the block and activation multicast
reuse. The default parameters preserve the conservative independent-position
model, so any claimed benefit must show sensitivity to reuse and controller
overhead rather than relying on free data movement.

The official GPU evidence adds a batching constraint to this design. For a
single request, an active-row MLP with explicit gather/scatter is no faster
than dense execution at block size 16. At batch 64, the same 16-to-9 row
reduction becomes useful. Consequently, the physical design should expose a
cross-request position-grouping buffer or persistent lane compactor; a
per-request sparse call is not an adequate implementation. This requirement
is DFlash-specific because block positions share a verification-prefix value
while requests can be grouped by their selected depth schedule.

For example, with an illustrative block of eight positions and depths
`(3,3,3,3,3,3,3,3)`, the model reports 22 cycles under one set of explicit
parameters. A staircase `(3,3,2,2,2,1,1,1)` reports 13 cycles under the same
parameters. These are oracle outputs, not measured hardware speedups; the
acceptance-preservation experiment and a calibrated cycle model are still
required.

## Main challenge

DFlash uses bidirectional attention inside a block. A tail position is low-value for commitment but may still provide context used by a prefix position. Naively skipping its computation can reduce prefix acceptance. The design space therefore includes:

1. shared lower layers plus selective upper layers;
2. frozen low-fidelity tail states used as context for prefix lanes;
3. delayed tail refinement after prefix lanes have reached a protected boundary;
4. full fallback when the predicted interference exceeds a safe threshold.

## Measurements

The minimum trace must contain request id, block size, accepted prefix length, draft position confidence, per-layer timing, and optional layer checkpoints. The first oracle reports:

- prefix survival curve and conditional hazard;
- confidence calibration by position;
- marginal survival gain per layer;
- compute saved by each staircase schedule;
- estimated routing, memory, and chiplet-link bytes.

## Baselines and novelty discipline

Compare against vanilla DFlash, fixed draft depth, dynamic draft length, D-Cut-style pruning, and any available budget-aware block-diffusion method. Do not call positional loss weighting, tree pruning, or draft-length control novel. A successful paper must demonstrate a hardware/dataflow mechanism that cannot be reduced to those baselines.

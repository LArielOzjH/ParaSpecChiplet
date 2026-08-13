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

## Cost-model boundary

The chiplet oracle models four terms separately: draft compute, activation-link
cycles, synchronization, and router/scheduler cycles. It can also account for
lower-layer execution shared once across the block and activation multicast
reuse. The default parameters preserve the conservative independent-position
model, so any claimed benefit must show sensitivity to reuse and controller
overhead rather than relying on free data movement.

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

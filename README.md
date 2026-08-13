# ParaSpecChiplet

Research workspace for architecture exploration of DFlash-style block-parallel speculative decoding.

## Current thesis under test

DFlash predicts a block in parallel, but the positions in that block do not have equal *survival value*: a mismatch at an early position invalidates the useful contribution of all later positions. Existing DFlash execution gives every position the same draft depth, precision, and data movement. ParaSpecChiplet studies whether a survival-aware, heterogeneous execution schedule can reduce draft cost while preserving target-model losslessness.

This is a hypothesis, not a result. The project will reject it if layer/position asymmetry does not survive real traces, if reduced-fidelity tail computation damages prefix acceptance, or if chiplet traffic removes the savings.

## Current research brief

The current DAC'27 recommendation is documented in
[`docs/dac27_research_brief.md`](docs/dac27_research_brief.md). The primary
hypothesis is SAGE-DFlash: state-conditioned block scheduling with a protected
prefix and heterogeneous upper-layer execution. Static protected-prefix
execution is the fallback; chiplets are an implementation option subject to a
traffic-aware kill gate.

The repository currently contains reproducible CPU activation probes, official
DFlash acceptance-trace adapters, state-conditioned survival analysis, schedule
frontier/controller oracles, and a chiplet cost model. Activation probes are
not acceptance results. The locally cached community 0.6B draft has an
incompatible/missing vocabulary mapping, so acceptance claims must come from
the official DFlash serving path or a checkpoint with valid mappings.

## Evidence boundary

- DFlash, [arXiv:2602.06036](https://arxiv.org/abs/2602.06036), already uses position-decayed training loss and reports a depth/acceptance/latency trade-off.
- D-PACE, [arXiv:2605.18810](https://arxiv.org/abs/2605.18810), studies dynamic position-aware training.
- D-Cut, [arXiv:2607.14647](https://arxiv.org/abs/2607.14647), and BASTION, [arXiv:2605.29727](https://arxiv.org/abs/2605.29727), study runtime budget/pruning decisions.

Therefore this repository does not claim novelty for positional weighting, dynamic draft length, or candidate pruning. The proposed contribution, if validated, is the architecture/dataflow: a layer-by-position survival map that routes early/high-value positions to full-fidelity resources and later/lower-value positions to cheaper resources while retaining enough shared computation to control bidirectional interference.

## Repository status

The first artifact is a dependency-free trace oracle. It computes prefix-survival curves, conditional failure hazards, and value-per-cost summaries from saved verification traces. It deliberately runs without a GPU so the research question can be falsified before simulator work.

## Planned workflow

1. Capture DFlash traces: draft block, target predictions, accepted prefix, per-layer confidence, and per-stage timings.
2. Measure `S_i = P(accepted_prefix >= i)` and the conditional hazard at each position.
3. Measure layer ablations or partial-depth outputs to estimate the layer-by-position value map.
4. Evaluate staircase schedules against uniform depth, fixed shallow depth, and runtime pruning baselines.
5. Only if the first gate passes, model heterogeneous chiplets and link traffic in a cycle/energy model.

## First-gate kill criteria

The primary idea is killed if any representative workload family shows one of these outcomes:

- no stable prefix-survival gradient across positions;
- later-position approximation changes early-position acceptance materially;
- the best heterogeneous schedule saves less than its routing/synchronization overhead;
- the schedule is equivalent to an already published dynamic draft-length or pruning method.

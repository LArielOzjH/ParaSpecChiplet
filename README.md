# ParaSpecChiplet

Research workspace for architecture exploration of DFlash-style block-parallel speculative decoding.

## Current thesis under test

DFlash predicts a block in parallel, but the positions in that block do not
have equal survival value, and its draft Transformer blocks do not have equal
marginal value for the committed prefix. ParaSpecChiplet studies an
acceptance-calibrated joint block-fidelity schedule: dense bidirectional
attention, heterogeneous MLP widths, cross-request grouping, and dense
fallback.

This is a hypothesis, not a result. The project will reject it if layer/position asymmetry does not survive real traces, if reduced-fidelity tail computation damages prefix acceptance, or if chiplet traffic removes the savings.

## Current research brief

The current DAC'27 recommendation is documented in
[`docs/dac27_final_direction.md`](docs/dac27_final_direction.md) and
[`docs/dac27_research_brief.md`](docs/dac27_research_brief.md). The primary
hypothesis is SAGE-DFlash: jointly validated block-fidelity scheduling on a
grouped monolithic engine. Position gating is a fallback; chiplets are an
implementation option subject to a traffic-aware kill gate.

The related-work boundary and the primary/fallback idea hierarchy are recorded
in [`docs/dac27_novelty_audit.md`](docs/dac27_novelty_audit.md) and
[`docs/dac27_idea_catalog.md`](docs/dac27_idea_catalog.md). The paper's novelty
anchor is the physical schedule/dataflow, not dynamic budget selection alone.
The current submission-level recommendation is summarized in
[`docs/dac27_recommendation.md`](docs/dac27_recommendation.md).

The repository currently contains official Qwen3-4B DFlash acceptance traces,
block/MLP/width ablation probes, joint schedule frontiers, RTX 4090 width
microbenchmarks, calibrated serving/cost models, and explicit chiplet
crossover studies. Python hooks and analytical models are not end-to-end
speedup results. The custom Triton prototype is recorded as a negative result.

## Evidence boundary

- DFlash, [arXiv:2602.06036](https://arxiv.org/abs/2602.06036), already uses position-decayed training loss and reports a depth/acceptance/latency trade-off.
- D-PACE, [arXiv:2605.18810](https://arxiv.org/abs/2605.18810), studies dynamic position-aware training.
- D-Cut, [arXiv:2607.14647](https://arxiv.org/abs/2607.14647), and BASTION, [arXiv:2605.29727](https://arxiv.org/abs/2605.29727), study runtime budget/pruning decisions.

Therefore this repository does not claim novelty for positional weighting, dynamic draft length, or candidate pruning. The proposed contribution, if validated, is the architecture/dataflow: a layer-by-position survival map that routes early/high-value positions to full-fidelity resources and later/lower-value positions to cheaper resources while retaining enough shared computation to control bidirectional interference.

## Repository status

The first artifact is a dependency-free trace oracle. It computes prefix-survival curves, conditional failure hazards, and value-per-cost summaries from saved verification traces. It deliberately runs without a GPU so the research question can be falsified before simulator work.

## Planned workflow

1. Capture official DFlash traces and measure `S_i = P(accepted_prefix >= i)`.
2. Measure joint block-fidelity schedules with the target verifier.
3. Calibrate width/row latency and construct a safe Pareto frontier.
4. Evaluate grouped monolithic execution with dense fallback and mixed queues.
5. Only if a physical organization adds value, model chiplets and link traffic.

## First-gate kill criteria

The primary idea is killed if any representative workload family shows one of these outcomes:

- no stable acceptance-safe block-fidelity schedule;
- joint fidelity loses early-prefix survival;
- fused/grouped execution saves less than fallback, queueing, or synchronization overhead;
- the schedule is equivalent to an already published dynamic draft-length or pruning method.

# ParaSpecChiplet

Research workspace for architecture exploration of DFlash-style block-parallel speculative decoding.

## Current thesis under test

DFlash predicts a block in parallel, but the positions in that block do not
have equal survival value, and its draft Transformer blocks do not have equal
marginal value for the committed prefix. ParaSpecChiplet studies an
acceptance-calibrated joint block-fidelity schedule: dense bidirectional
attention, heterogeneous MLP widths, cross-request grouping, and dense
fallback.

The current paper scope is an acceptance-calibrated architecture/dataflow and
cost study. It does not claim end-to-end serving speedup: the generic Triton
kernel failed the latency gate, and chiplets remain conditional.

## Current research brief

The current DAC'27 recommendation is documented in
[`docs/dac27_final_direction.md`](docs/dac27_final_direction.md) and
[`docs/dac27_research_brief.md`](docs/dac27_research_brief.md). The primary
hypothesis is SAGE-DFlash: jointly validated position/depth MLP scheduling on
a grouped monolithic engine. Chiplets are an implementation option subject to
a traffic-aware break-even gate.

For the shortest paper-ready summary, see
[`docs/dac27_paper_pitch.md`](docs/dac27_paper_pitch.md). The authoritative
claim/evidence table is
[`docs/dac27_evidence_matrix.md`](docs/dac27_evidence_matrix.md), and the
related-work boundary is in
[`docs/literature_positioning_2026.md`](docs/literature_positioning_2026.md).

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

## Completed research workflow

1. Captured official DFlash traces and measured `S_i = P(accepted_prefix >= i)`.
2. Measured joint block-fidelity schedules with the target verifier.
3. Calibrated width/row latency and constructed acceptance/cost frontiers.
4. Integrated grouped monolithic execution with dense fallback and mixed queues.
5. Modeled chiplet traffic and quantified its break-even boundary.
6. Re-ran the fixed-shape Triton kernel at batch 1/8/64 and recorded the
   negative result.

## Final claim boundary

The current submission explicitly does not claim:

- end-to-end serving speedup;
- universal acceptance safety for static block-width rankings;
- a positive chiplet advantage;
- a correctness-passing or faster generic Triton fused kernel.

The proposed contribution is the measured, acceptance-constrained dataflow and
its explicit occupancy/traffic cost boundary.

# Candidate DAC'27 Ideas

## Current evidence-based decision

The strongest current paper core is **survival-aware grouped DFlash execution**:
dense bidirectional attention, protected-prefix upper-layer MLP gating, and a
batch-aware grouped execution engine with dense fallback. The official serving
loop supports acceptance compatibility for the protected schedules, while the
RTX 4090 measurements support savings only in sufficiently large batches.

State-conditioned scheduling remains an optional extension, not the primary
claim: its current evidence is descriptive and confounded by prompt and decode
phase. Chiplets likewise remain an optional physical realization; the current
equal-resource and small-batch evidence does not justify leading with them.

These are research candidates, not claims. Each idea has a measurable kill condition.

## A. Survival-aware grouped DFlash (primary)

**Motivation.** A DFlash block is verified as a prefix. Later positions are conditionally useful, but the draft accelerator spends the same depth and precision on every position.

**Challenge.** Bidirectional attention couples tail positions back into prefix hidden states. Position pruning or layer skipping can reduce the very prefix acceptance that the optimization is meant to protect.

**Mechanism.** Keep shared lower-layer execution and dense bidirectional
attention, then route position rows through a layer-by-position staircase:
protected prefix lanes receive full upper-layer MLP execution; tail lanes
receive reduced MLP depth. A calibrated dense fallback bypasses compaction when
the batch or active-row group is too small. Compare the grouped monolithic
engine against an equal-resource chiplet realization.

**Evidence needed.** `S_i=P(A>=i)`, conditional hazard, layer ablation gain, cross-position interference, and bandwidth/traffic-aware cycle estimates.

**Kill condition.** Tail approximation reduces early-prefix acceptance enough
to erase compute savings, or calibrated grouping yields no benefit in the
target throughput-serving regime.

## B. Verifier-feedback refinement pipeline

**Motivation.** The target verifier produces acceptance information for free after each cycle. A future draft cycle can use this feedback to decide which position/layer groups deserve refinement rather than applying a fixed staircase.

**Challenge.** Feedback arrives after the current block is verified, while the next DFlash block must remain low-latency. The controller must be cheap and robust to workload phase changes.

**Mechanism.** Maintain a small per-request survival state: recent
accepted-prefix histogram, confidence calibration, and layer-value estimate.
Use it to select one of several precompiled heterogeneous schedules. This is
schedule selection, not token pruning, and is only an extension to Idea A.

**Evidence needed.** Adaptation overhead, phase-change recovery, tail latency, and comparison against D-Cut/Bastion-style budget policies.

**Kill condition.** A static schedule matches the adaptive schedule, or controller overhead exceeds the saved draft work.

## C. Layer-specialized DFlash chiplet fabric

**Motivation.** DFlash's target hidden features are injected into every draft layer. Different draft depths have different acceptance/latency trade-offs, and newer DFlare-style drafters increase per-layer conditioning expressiveness. A monolithic accelerator cannot efficiently serve all depth/position regimes simultaneously.

**Challenge.** Layer specialization can create activation movement and synchronization costs that outweigh reuse. The fabric also must support requests with different draft depths under continuous batching.

**Mechanism.** Build reusable layer chiplets grouped by fidelity/cost class, with activation multicast for shared lower layers and elastic routing for upper layers. Map the survival-aware schedule from A onto this fabric.

**Evidence needed.** Area-normalized throughput, activation/link bytes, queueing under mixed depth requests, and energy per committed token.

**Kill condition.** A monolithic accelerator with the same total resources wins across all realistic batch/context regimes.

## Ranking

| Candidate | Novelty | Architecture depth | Main risk | Decision |
|---|---:|---:|---|---|
| A | high if hardware-only claim holds | high | bidirectional interference | profile first |
| B | medium-high | medium-high | may look like runtime policy | keep as extension |
| C | medium | high | chiplet becomes cosmetic | use only with A evidence |

The paper should not lead with “chiplets are good.” The lead should be the unequal survival value and the execution mismatch; chiplets are one physical organization that makes the proposed heterogeneity practical.

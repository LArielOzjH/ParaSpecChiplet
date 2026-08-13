# Candidate DAC'27 Ideas

## Current evidence-based decision

The strongest current paper hypothesis is **block-importance-aware DFlash
execution**: keep bidirectional attention dense, assign heterogeneous MLP
fidelity to draft Transformer blocks according to their marginal contribution
to prefix acceptance, and use grouped execution with dense fallback. Official
Qwen3-4B ablations show strong non-uniformity: bypassing block 0's MLP reduces
`S1` from `0.6095` to `0.3592`, while bypassing block 2's MLP preserves `S1`
(`0.6096`) and nearly preserves `S2` (`0.3307` versus `0.3368`).

This is acceptance evidence, not yet a speedup result: the Python probe still
executes the MLP before zeroing its output. The next gate is a fused
block-heterogeneous MLP implementation and an equal-resource monolithic
baseline. A composition stress test further shows that single-block scores do
not compose independently: bypassing MLP blocks `{2,3,4}` lowers mean accepted
prefix to `0.7692`. The scheduler therefore needs joint-schedule awareness,
not only a per-block ranking. Partial MLP fidelity is a more promising
primitive: scaling layers 2+3 to `alpha=0.5` preserved the measured early
prefix better than zeroing them, although this remains an acceptance-only
probe because the MLP is still executed. A reduced-width acceptance probe
further preserved layer 2 at 50% intermediate width (`S1=0.6103`), while the
same width reduction on layers 2+3 fell to `S1=0.5625`; the hardware mechanism
therefore needs a jointly validated schedule table.

State-conditioned scheduling remains an optional extension, not the primary
claim: its current evidence is descriptive and confounded by prompt and decode
phase. Chiplets likewise remain an optional physical realization; the current
equal-resource and small-batch evidence does not justify leading with them.

These are research candidates, not claims. Each idea has a measurable kill condition.

## A. Draft-block importance-aware DFlash (primary candidate)

**Motivation.** DFlash injects target hidden features through a stack of draft
Transformer blocks, but the official ablation shows that those blocks do not
contribute equally to the accepted prefix. Uniform layer execution therefore
wastes work on low-value block updates while under-provisioning the blocks that
protect early acceptance.

**Challenge.** Full-block skipping is unsafe: bypassing any layer lowers
acceptance, and bypassing the first layer is catastrophic. The architecture
must preserve dense cross-position attention, selectively reduce only safe
updates, and account for activation movement and queueing overhead.

**Mechanism.** Keep attention dense in every block, classify block MLP updates
by measured marginal prefix value, and route high-value blocks to full-fidelity
lanes while low-value blocks use reduced precision or a fused skipped MLP
update. Group requests with compatible block schedules and fall back to dense
monolithic execution when the group is too small. A chiplet fabric is an
optional physical realization for shared and specialized MLP lanes.

**Evidence needed.** Per-block `S_i` degradation, held-out stability of the
ranking, MLP-only acceptance, fused MLP latency, activation bytes, queueing,
and an equal-resource monolithic comparison.

**Kill condition.** No block-level MLP fidelity schedule preserves early
prefix acceptance, or fused grouped execution loses to dense monolithic
execution after movement and synchronization costs.

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

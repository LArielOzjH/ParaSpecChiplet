# DAC'27 Paper Pitch: SAGE-DFlash

## Working title

**SAGE-DFlash: Prefix-Survival-Aware MLP Scheduling for Block-Parallel
Speculative Decoding**

## Abstract draft

Block-parallel speculative decoders such as DFlash compute many draft tokens
simultaneously, but target verification commits only a contiguous prefix. This
creates a hardware mismatch: draft positions with different probabilities of
survival receive the same MLP computation, while bidirectional attention makes
naive tail deletion unsafe. SAGE-DFlash introduces an acceptance-calibrated
MLP scheduling dataflow that keeps attention dense and assigns a jointly
validated position/depth schedule to each draft block. Compatible requests are
coalesced into grouped MLP execution, and an occupancy estimator selects dense
fallback when compaction is not profitable. On Qwen3-4B/DFlash-b16, official
verification traces show non-uniform block value and preserve the protected8
candidate's early-prefix survival while removing 25% of nominal MLP rows. RTX
4090 measurements expose the throughput regime and the dense/grouped crossover.
We report the resulting acceptance, traffic, queue, and cost frontiers against
equal-resource monolithic and conditional chiplet organizations. The study
also reports negative results for static block-width ranking, simple
state-threshold adaptation, and a generic Triton fused kernel, clarifying the
limits of hardware speedup claims.

## Motivation in three sentences

1. DFlash is parallel in draft generation but serial in what verification can
   commit: only the accepted prefix creates useful speculative work.
2. Official ablations show that block value is non-uniform, but held-out tests
   show that a fixed “important layer” ranking is not stable.
3. The architecture problem is therefore to schedule *joint fidelity* under a
   survival constraint while paying real compaction, queue, and movement cost.

## Core challenge

The schedule cannot simply skip tail tokens. DFlash attention is bidirectional
within the block, so tail hidden states can affect protected prefix states.
MLP updates are position-wise and provide the safer control point, but their
gather/scatter cost is workload- and occupancy-dependent. The design must also
avoid treating individually safe block reductions as independently composable.

## Three contributions

### 1. Acceptance-calibrated schedule abstraction

Represent a candidate by a finite position/depth vector and register it only
after joint target-verifier evaluation. Use prefix survival `S_i=P(A>=i)` as
the primary value metric; use hidden-state distance only as a screening signal.

### 2. Occupancy-aware grouped MLP dataflow

Keep attention dense, group compatible MLP rows across requests, and select
dense fallback when active-row occupancy is below the measured crossover. The
architecture is meaningful with non-ideal arrival mixtures rather than only
under a zero-overhead sparse-MAC oracle.

### 3. Equal-resource boundary study

Compare grouped monolithic execution with a traffic-aware chiplet realization.
Chiplets are promoted only if specialization/utilization or area/energy
sharing exceeds the explicit link/router/synchronization break-even boundary.
The paper reports the failed generic Triton kernel and static-width schedule
as negative results, preventing an idealized speedup claim.

## Evaluation matrix

| Dimension | Baselines / cases | What it proves |
|---|---|---|
| Acceptance | uniform, protected schedules, held-out workloads | fidelity safety and generality |
| Dataflow | dense, eager grouped, dense fallback | compaction policy behavior |
| Batch | 1, 8, 64 | serving regime and low-batch limit |
| Mixture | homogeneous, 25/50/75% schedule classes | queue coalescing sensitivity |
| Policy | static table, offline oracle, state selector | policy value separated from datapath |
| Fabric | monolithic, grouped monolithic, chiplet | traffic/sync/area boundary |
| Kernel | eager reduced, Triton prototype | implementation limit, not ideal speedup |

## Current evidence snapshot

- Protected8 held-out acceptance with fallback: mean prefix `1.4371` vs
  `1.4219` uniform; `S1=0.6478` vs `0.6375`.
- Nominal MLP rows: `60/80` for protected8 staircase.
- CUDA-graph grouped MLP, batch 64, 9 active rows/request: `680.02 ms` vs
  `1038.22 ms` dense for one layer.
- Batch 1/8 selective row grouping: no reliable payback.
- Chiplet break-even: approximately `1.005–1.021×` effective gain depending on
  link bandwidth and width.
- Generic Triton kernel: `2.72–5.00×` slower than eager reduced MLP across
  batch 1/8/64.

All positive hardware numbers above are either isolated microbenchmarks or
analytical/calibrated proxies. The current submission must not call them
end-to-end serving throughput.

## Reviewer-risk answers

**“Is this just token pruning or early exit?”**  No. Attention remains dense;
the scheduled unit is MLP fidelity under a target-verifier prefix-survival
constraint, with explicit bidirectional-context and queue costs.

**“Why not always reduce the same layer?”**  Held-out block-width experiments
reject that rule; joint schedules are measured and filtered rather than ranked
independently.

**“Why chiplets?”**  Chiplets are not assumed to help. They are an optional
physical mapping evaluated against an equal-resource monolithic baseline and a
quantified break-even condition.

**“Where is the speedup?”**  This version makes no end-to-end speedup claim.
It contributes an acceptance-calibrated dataflow and a transparent cost
boundary; a future tensor-core-aware persistent kernel is required to reopen
the speedup claim.

## Claim-safe conclusion

SAGE-DFlash establishes that DFlash exposes a real mismatch between uniform
draft computation and prefix commitment, and turns that mismatch into a
measurable, acceptance-constrained execution/dataflow design. It does not
claim that every workload benefits, that static block specialization is safe,
or that chiplets and generic fused kernels automatically improve throughput.

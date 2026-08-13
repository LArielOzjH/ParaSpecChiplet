# DAC'27 Novelty Audit and Idea Hierarchy

Checked against the current arXiv records on 2026-08-14. This note is a
submission aid, not a claim that the listed papers have been exhaustively
reviewed.

## The novelty boundary

The paper must not be framed as another policy that observes confidence or
recent acceptance and chooses a cheaper speculative budget. That abstraction
is already occupied by dynamic-length, pruning, and repair methods. The
architecture contribution must remain meaningful when the schedule selector is
replaced by an offline oracle.

The defensible boundary is:

> DFlash exposes a prefix-survival value map over a block-parallel draft
> computation. SAGE-DFlash turns that map into a physical heterogeneous
> execution dataflow: dense bidirectional attention, jointly calibrated
> position/depth MLP schedules, cross-request schedule grouping, and a
> measured dense fallback.

This makes the central object a schedule/dataflow pair, not a confidence
heuristic.

## Closest neighbors

| Work | What it establishes | What SAGE-DFlash must not claim | Remaining boundary |
|---|---|---|---|
| [DFlash](https://arxiv.org/abs/2602.06036) | Block-parallel diffusion drafting with target hidden-feature conditioning | Novelty of block drafting or target conditioning | Physical execution under unequal committed-prefix value |
| [D-Cut](https://arxiv.org/abs/2607.14647) | Cross-request adaptive verification-depth pruning and environment-aware cost modeling | Generic batch-aware pruning or cost-aware budget selection | Preserve dense draft positions while changing internal block fidelity |
| [LibraSpec](https://arxiv.org/abs/2608.08721) | Marginal-gain optimization for dynamic speculative length | Expected-speedup optimization or dynamic length selection | The selected schedule must map to heterogeneous lanes and queue policy |
| [CURE](https://arxiv.org/abs/2608.00531) | Local uncertainty repair for block-parallel drafting | Tail repair, uncertainty routing, or accepted-length improvement alone | Avoid repair-tree semantics; specialize the original DFlash datapath |
| [Compass](https://arxiv.org/abs/2512.06093) | General heterogeneous chiplet mapping for dynamic LLM services | Chiplets or mapping search as novelty by themselves | DFlash-specific prefix survival, dense attention coupling, and MLP grouping |

## Idea hierarchy

### 1. Primary: SAGE-DFlash grouped fidelity engine

**Motivation.** Verification commits a prefix, while the draft stack spends
roughly uniform MLP work across block positions and layers. Official probes
show unequal block value and show that individually safe reductions do not
compose independently.

**Mechanism.** Keep attention dense for every block position. Select a jointly
validated position/depth MLP schedule, group requests with compatible vectors,
and choose dense execution when the active group is too small to amortize
compaction.

**Challenge.** Tail states still participate in bidirectional context, and the
schedule has to account for queueing, movement, synchronization, and fallback
rather than only MAC count.

**Claim if the gate passes.** An acceptance-calibrated heterogeneous dataflow
improves committed-token cost in a throughput-serving regime against an
equal-resource monolithic dense baseline. The selector must be evaluated in
three forms: static uniform, offline oracle, and state/workload-conditioned
runtime selection.

**Kill condition.** No joint schedule survives held-out acceptance calibration,
or grouped execution cannot beat dense fallback after measured overheads.

### 2. Conservative baseline/fallback: static fidelity schedule table

**Motivation.** State-conditioned policies may add little predictive value and
are close to existing dynamic-budget methods. A static table is therefore the
required conservative baseline, not an assumed success case.

**Mechanism.** Offline-select a small table of joint width vectors from held-out
official traces. Hardware only performs table lookup, grouping, and fallback.

**Challenge.** The schedule must transfer across prompt families and avoid
overfitting the 12-prompt calibration trace.

**Claim if it survives held-out calibration.** The contribution is a regular,
oracle-schedulable multi-fidelity MLP datapath, even if runtime adaptation is
removed.

**Kill condition.** The Pareto frontier collapses to uniform execution on
held-out workloads.

### 3. Optional physical extension: elastic heterogeneous/chiplet fabric

**Motivation.** Full- and reduced-width lanes may have different utilization
patterns; physical separation could provide area sharing or workload
isolation.

**Mechanism.** Map shared dense attention/backbone and heterogeneous MLP lanes
to physical regions or chiplets with activation multicast and explicit
synchronization.

**Challenge.** Link traffic and barriers can exceed the saved MLP work. The
comparison must use equal total resources and include a monolithic grouped
baseline.

**Claim.** Only a conditional mapping result: chiplets help in a measured
regime with sufficient reuse and group occupancy.

**Kill condition.** The current analytical sweep already suggests this risk:
chiplet cost is above equal-resource monolithic cost across tested bandwidths.
If calibrated measurements agree, remove chiplets from the title and headline.

## Reviewer attack / answer

**“Is this just token pruning?”** No. The primary design keeps every draft
position in dense attention and changes the internal MLP fidelity of a joint
block schedule. A token-pruning baseline must be measured, but it does not
capture bidirectional-context preservation.

**“Is this just dynamic speculation length?”** No. The schedule can be supplied
by an offline oracle for the architecture experiment; the contribution is the
physical execution substrate. Runtime adaptation is a gate for practical
selection, not the novelty anchor.

**“Why does block importance matter if one layer is safe?”** Because the
interaction is non-additive. The observed `{2,3}` schedule loses early-prefix
survival even though single-layer reductions can pass. This motivates joint
frontier measurement rather than independent ranking.

**“Why should a chiplet be necessary?”** It is not necessary for the minimum
paper. Chiplets are retained only as a calibrated physical extension; the
monolithic grouped engine is the required baseline and the likely fallback.

## Minimum publishable result

The paper remains viable without a positive chiplet result and without a
low-batch kernel speedup if it provides:

1. official target-verifier evidence for a held-out joint fidelity frontier;
2. a concrete grouped monolithic dataflow with dense fallback;
3. an equal-resource cost model calibrated with measured row/width latency;
4. explicit baseline comparisons against token pruning/dynamic-budget methods,
   plus a chiplet crossover study that reports negative regimes when it loses.

End-to-end throughput must be claimed only after a correctness-tested fused or
persistent implementation wins in the stated batch regime.

# DAC'27 Direction Audit

## Recommended primary paper

**SAGE-DFlash: Prefix-Survival-Aware MLP Scheduling for Block-Parallel
Speculative Decoding**

The minimum defensible contribution is an acceptance-calibrated scheduler and
grouped monolithic dataflow:

1. keep bidirectional attention dense;
2. assign a finite position/depth MLP schedule across each draft block;
3. validate schedules jointly with official target verification;
4. use measured width/row latency to remove dominated schedules;
5. group compatible requests and use dense fallback when grouping does not
   pay.

The original 12-prompt calibration found layer 2 at 50% MLP width to be a
low-regret point (`S1=0.6103` versus `0.6095` for uniform), while a two-layer
50% schedule fell to `S1=0.5625`. A new 8-prompt held-out screening run makes
the boundary sharper: layer 2 at 50% gives mean prefix `0.9624` versus `1.0234`
for uniform, and layers 2+3 gives `0.8561`. Thus layer 2 is evidence of
unequal block value, not a universally safe static schedule. Joint schedule
validation and workload/state conditioning are essential.

A held-out position-schedule run with 96 generated tokens (8 workload-diverse
prompts) kept the protected8 staircase schedule at or above uniform: mean
accepted prefix `1.4448` versus `1.4219`, `S1=0.6530` versus `0.6375`, and
`S4=0.0946` versus `0.0938`. This upgrades it from a short-trace screening
candidate to the primary acceptance-compatible schedule, while still not
establishing speedup. The held-out block-width result is more cautionary; see
[`docs/heldout_schedule_results.md`](heldout_schedule_results.md) and
[`docs/heldout_block_width_results.md`](heldout_block_width_results.md).
The evidence comparison and the resulting primary-idea decision are summarized
in [`docs/primary_idea_decision.md`](primary_idea_decision.md): position/depth
scheduling is currently stronger than static layer-width specialization.

## Claims currently supported

- draft blocks and positions have unequal marginal value;
- full-block bypass is unsafe, especially for the first block;
- dense attention plus selected position/depth MLP fidelity is substantially
  safer than static layer-width reduction;
- reduced-width MLP has real GPU compute-side headroom at throughput batch;
- a measured acceptance/latency frontier can select safe schedules;
- chiplet links and synchronization lose in the current equal-resource model.

## Claims not supported

- end-to-end serving speedup;
- low-batch benefit;
- a correctness-passing custom Triton fused kernel;
- chiplet advantage;
- generalized state-aware policy improvement (only a preliminary 8-prompt
  threshold probe is positive);
- arbitrary composition of individually safe block schedules.

## Architecture story

The proposed hardware is a monolithic grouped-fidelity engine: dense
attention lanes feed a schedule-aware, row-selective MLP fabric with full and
reduced-depth lanes,
cross-request grouping, a small schedule table, and dense fallback. The
controller is not a generic token-pruning controller; it selects a jointly
validated block-fidelity vector under a prefix-survival constraint.

The fabric also needs an occupancy gate. Under the current normalized
launch/scatter calibration, the queue sweep crosses from grouped loss to
grouped benefit near 40--50% heterogeneous-schedule occupancy. The controller
must estimate this occupancy and choose dense fallback rather than always
invoke a selective path. See
[`docs/reconfigurable_fabric_sweep.md`](reconfigurable_fabric_sweep.md).

A preliminary state-conditioned width probe was positive at 32 tokens, but the
96-token follow-up rejected the simple previous-prefix threshold; see
[`docs/state_conditioned_width_results.md`](state_conditioned_width_results.md).

Chiplets are an optional extension only if a later design demonstrates that
parallel specialized lanes, area/energy sharing, or workload isolation
overcomes link and synchronization costs. The current analytical sweep does
not demonstrate that condition.

## Remaining decisive gate

Implement a correctness-tested tensor-core-aware reduced-width kernel with
fixed shapes and compare it against the existing eager reduced-width path at
batch 1, 8, and 64. If it does not win at batch 64, remove the hardware
speedup claim and present the work as an acceptance-calibrated architecture
and cost study. If it wins only at batch 64, explicitly scope the paper to
throughput-serving.

This decision keeps the paper viable without relying on grammar, sparse head,
chiplet optimism, or an unverified compiler/kernel result.

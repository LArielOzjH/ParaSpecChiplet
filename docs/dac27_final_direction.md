# DAC'27 Direction Audit

## Recommended primary paper

**SAGE-DFlash: Joint Fidelity Scheduling for Block-Parallel Speculative
Decoding**

The minimum defensible contribution is an acceptance-calibrated scheduler and
grouped monolithic dataflow:

1. keep bidirectional attention dense;
2. assign a finite MLP fidelity vector across draft blocks;
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

A small held-out position-schedule screening run (8 workload-diverse prompts,
32 generated tokens) kept the protected8 staircase schedule close to uniform:
mean accepted prefix `1.0312` versus `1.0234`, `S1=0.5547` versus `0.5312`, and
identical `S4=0.0469`. Prompt-level intervals are wide, so this supports
acceptance-compatibility screening rather than a generalized gain claim. The
held-out block-width result is more cautionary; see
[`docs/heldout_schedule_results.md`](heldout_schedule_results.md) and
[`docs/heldout_block_width_results.md`](heldout_block_width_results.md).

## Claims currently supported

- draft blocks have unequal marginal value;
- full-block bypass is unsafe, especially for the first block;
- dense attention plus selected MLP fidelity is substantially safer;
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
attention lanes feed a schedule-aware MLP fabric with full/half-width lanes,
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

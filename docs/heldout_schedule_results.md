# Held-Out Joint Schedule Screening

Date: 2026-08-14

This is a small cross-workload screening experiment for the acceptance
compatibility of the current protected-prefix schedules. It is not an
end-to-end speedup result and is not large enough to establish statistical
generalization.

## Setup

- Target: Qwen3-4B
- Draft: Qwen3-4B-DFlash-b16
- GPU: RTX 4090
- Prompts: 8 held-out coding, math, networking, systems, database, and
  accelerator questions
- Schedules: uniform and protected8 staircase (the probe also recorded the
  two existing stress schedules)
- Generation budget: 32 tokens per prompt/schedule
- Target verifier and DFlash loop: unchanged
- Attention: dense; only the MLP row schedule was gated

Raw events are in [`data/heldout_mlp_gating.jsonl`](../data/heldout_mlp_gating.jsonl)
and prompts are in [`data/heldout_workload_prompts.txt`](../data/heldout_workload_prompts.txt).

## Aggregate result

| Schedule | Events | Mean accepted prefix | `S1` | `S2` | `S4` |
|---|---:|---:|---:|---:|---:|
| Uniform | 128 | 1.0234 | 0.5312 | 0.2734 | 0.0469 |
| Protected8 staircase | 128 | 1.0312 | 0.5547 | 0.2656 | 0.0469 |

The staircase schedule has a small positive aggregate change in mean prefix
and `S1`, a small negative change in `S2`, and no change in `S4`. Prompt-level
bootstrap intervals over the eight prompts are wide: approximately
`[-0.022, 0.035]` for mean-prefix delta, `[-0.008, 0.069]` for `S1` delta, and
`[-0.018, 0.000]` for `S2` delta. These intervals are descriptive only.

## Interpretation

1. The schedule remains a plausible acceptance-compatible candidate outside
   the original 12-prompt calibration set.
2. The result does not establish a universal acceptance gain or a best
   schedule. The unchanged `S4` and the wide prompt-level intervals argue for
   a held-out frontier rather than a single fixed schedule claim.
3. The main paper should report the schedule as a safe/low-regret point on a
   measured frontier, then compare several jointly calibrated vectors.
4. A larger experiment should use matched prompt splits, longer generation,
   multiple random seeds where nondeterminism matters, and explicit token
   pruning/dynamic-budget baselines.

The correct current claim is therefore **cross-workload acceptance
compatibility screening**, not generalized speedup or schedule optimality.

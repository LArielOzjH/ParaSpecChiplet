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

## 96-token follow-up

The same eight prompts and schedules were rerun with a 96-token budget. Raw
events are in
[`data/heldout_mlp_gating_96.jsonl`](../data/heldout_mlp_gating_96.jsonl).

| Schedule | Events | Mean accepted prefix | `S1` | `S2` | `S4` | MLP work |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 320 | 1.4219 | 0.6375 | 0.3781 | 0.0938 | 80 |
| Protected8 conservative | 318 | 1.4371 | 0.6447 | 0.3836 | 0.0943 | 67 |
| Protected8 staircase | 317 | 1.4448 | 0.6530 | 0.3817 | 0.0946 | 60 |
| Protected4 moderate | 322 | 1.4068 | 0.6429 | 0.3820 | 0.0963 | 51 |

The protected8 staircase remains acceptance-compatible at the longer budget,
with a small positive change in every reported survival metric. At the
prompt-level view, six prompts match uniform and two improve; none shows a
negative change in this run. This is stronger evidence than the 32-token
screening, but it remains a Python row-gating experiment rather than a
latency result.

With the measured 38% MLP fraction, the staircase corresponds to a normalized
dense-attention plus MLP work fraction of `0.905`, before gather/scatter,
launch, queueing, and verification costs. The resulting committed-prefix per
work proxy is `1.5965` versus `1.4219` for uniform. This is a calibrated work
proxy, not an end-to-end speedup claim.

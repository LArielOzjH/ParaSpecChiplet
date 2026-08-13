# Held-Out Block-Width Acceptance Results

Date: 2026-08-14

This experiment tests whether the apparent block/layer importance survives a
new prompt set. It uses the official Qwen3-4B/DFlash-b16 serving loop and
truncated MLP widths; it is acceptance evidence, not a hardware speedup.

## Setup

- RTX 4090
- 8 held-out coding, math, networking, systems, database, and accelerator
  prompts
- 32 generated tokens per prompt and schedule
- dense attention and unchanged target verifier
- uniform MLP width versus one or two draft layers at 50% intermediate width

Raw traces:

- [`data/heldout_width_l2_05.jsonl`](../data/heldout_width_l2_05.jsonl)
- [`data/heldout_width_l3_05.jsonl`](../data/heldout_width_l3_05.jsonl)
- [`data/heldout_width_l23_05.jsonl`](../data/heldout_width_l23_05.jsonl)

## Aggregate result

The uniform control was rerun in each probe and produced the same aggregate
summary. The reduced-width schedules were:

| Schedule | Mean accepted prefix | `S1` | `S2` | `S4` |
|---|---:|---:|---:|---:|
| Uniform | 1.0234 | 0.5312 | 0.2734 | 0.0469 |
| Layer 2 at 50% | 0.9624 | 0.5263 | 0.2406 | 0.0526 |
| Layer 3 at 50% | 0.8696 | 0.4710 | 0.2391 | 0.0507 |
| Layers 2+3 at 50% | 0.8561 | 0.4892 | 0.2518 | 0.0144 |

## What this changes

1. Block/layer importance is real but is not a stable static ranking. Layer 2
   is less damaging than layer 3 on this workload, but its aggregate reduction
   still loses mean prefix and `S2`.
2. The joint layer-2+3 schedule is worse than either single-layer schedule,
   reinforcing the non-additivity observed on the original prompt set.
3. Prompt-level behavior changes direction: layer 2 helps prompts 0, 1, 2,
   and 4, but hurts prompts 3, 5, 6, and 7; layer 3 is preferable on some of
   the latter prompts. The small sample and short generation budget make this
   descriptive, not causal.
4. A fixed schedule table selected from one workload is not yet a publishable
   main claim. The architecture must either condition schedule selection on an
   observable entering state/workload signal, or retain uniform execution when
   the signal is uncertain.

## Revised research implication

The strongest surviving idea is a **reconfigurable block-fidelity fabric**:

- schedule vectors are first-class hardware descriptors;
- multiple full/reduced-width MLP modes are supported;
- an acceptance-calibrated selector chooses among jointly measured vectors;
- dense fallback is mandatory for unseen or low-confidence states;
- the physical contribution is grouped execution of the selected vector, not
  the selector alone.

This result is a warning against a static “layer 2 is safe” story. The next
gate is a held-out state-conditioned selector with an explicit static-uniform
baseline and controller overhead. If it cannot beat static uniform plus dense
fallback, the heterogeneous fidelity claim should be killed rather than
rescued by more aggressive width reductions.

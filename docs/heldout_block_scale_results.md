# Held-Out Block-Fidelity Scale Results

Date: 2026-08-14

## Purpose

The block-ablation results show that draft blocks have unequal marginal value,
but a single workload is not enough to justify a static block specialization.
This experiment tests whether scaling the MLP update of one or more blocks to
50% is acceptance-compatible on the held-out workload.

Attention remains dense. The scale hook is an acceptance probe: it executes
the Python MLP and scales its output, so these numbers are not hardware
latency or speedup measurements.

## Setup

- Target: Qwen3-4B
- Draft: Qwen3-4B-DFlash-b16
- Draft depth: 5 Transformer blocks
- Block size: 16
- GPU: RTX 4090
- Workload: 8 held-out prompts, 96 generated tokens
- Scale: selected MLP output multiplied by `0.5`

## Results

| Schedule | Mean accepted prefix | `S1` | `S2` | `S4` |
|---|---:|---:|---:|---:|
| Uniform | 1.4219 | 0.6375 | 0.3781 | 0.0938 |
| Block 2 MLP ×0.5 | 1.3720 | 0.6250 | 0.3598 | 0.0915 |
| Block 3 MLP ×0.5 | 1.3474 | 0.6073 | 0.3776 | 0.0876 |
| Blocks 2+3 MLP ×0.5 | 1.3636 | 0.6212 | 0.3758 | 0.0879 |

The joint schedule is not rescued by combining individually plausible blocks;
all three reduced-fidelity schedules are below uniform on this held-out set.
This contradicts a naive static rule such as “always reduce block 2.”

## Architectural consequence

The evidence still supports unequal block importance as the motivation, but
not static block specialization as the mechanism. Block value is conditioned
by workload, position, and interactions with other blocks. Therefore:

1. use block ablation and fidelity sweeps to construct candidate schedules;
2. validate schedules jointly against a prefix-survival constraint;
3. keep a robust position/depth schedule as the primary datapath candidate;
4. use dense fallback whenever no registered schedule is safe or grouping does
   not pay.

This is a negative result for a simple block-specific width/scale fabric, not
a negative result for block-aware scheduling in general. A future positive
block-level mechanism would need finer fidelity choices, richer state
features, or an explicitly conservative joint schedule table.

Raw traces:

- [`data/heldout_scale_l2_05.jsonl`](../data/heldout_scale_l2_05.jsonl)
- [`data/heldout_scale_l3_05.jsonl`](../data/heldout_scale_l3_05.jsonl)
- [`data/heldout_scale_l23_05.jsonl`](../data/heldout_scale_l23_05.jsonl)

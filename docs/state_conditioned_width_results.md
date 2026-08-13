# Preliminary State-Conditioned Width Probe

Date: 2026-08-14

This probe tests whether the previous cycle's accepted prefix can select a
reduced-width MLP mode without the fixed-layer failure seen in the held-out
static experiment. The target verifier is unchanged. The probe duplicates the
official DFlash generation loop only to update the MLP-width hook between
cycles; it is not a latency implementation.

## Setup

- Target: Qwen3-4B
- Draft: Qwen3-4B-DFlash-b16
- GPU: RTX 4090
- 8 held-out workload prompts
- 32 generated-token budget
- state rule: use layer 2 at 50% width when the previous accepted prefix is at
  least the threshold; otherwise use uniform width
- thresholds: 1 and 2
- strict control: the same custom loop with threshold 999, which never
  selects reduced width

Raw traces:

- [`data/heldout_state_width_uniform_control.jsonl`](../data/heldout_state_width_uniform_control.jsonl)
- [`data/heldout_state_width_t1.jsonl`](../data/heldout_state_width_t1.jsonl)
- [`data/heldout_state_width_t2.jsonl`](../data/heldout_state_width_t2.jsonl)

## Aggregate result

| Policy | Reduced-width cycles | Events | Mean prefix | `S1` | `S2` | `S4` |
|---|---:|---:|---:|---:|---:|---:|
| Uniform control | 0 | 128 | 1.0234 | 0.5312 | 0.2734 | 0.0469 |
| Threshold 1 | 66 | 128 | 1.0391 | 0.5625 | 0.2734 | 0.0547 |
| Threshold 2 | 33 | 127 | 1.0551 | 0.5669 | 0.2835 | 0.0551 |

The strict custom-loop control exactly matches the earlier held-out uniform
aggregate, which is a useful implementation sanity check. Both selectors show
a small positive acceptance change on this short workload, while threshold 2
uses half as many reduced-width cycles as threshold 1.

## Interpretation and limits

1. This is the first causal screening result for a state-conditioned schedule:
   changing the schedule after verification changes subsequent acceptance
   behavior, and a conservative threshold can avoid some static-schedule
   failures.
2. The effect is not uniformly positive by prompt. Threshold 1 improves the
   first three/four workload instances but hurts others; threshold 2 avoids
   several changes because it selects reduced width less often, but still has
   prompt-dependent behavior.
3. The experiment does not include selector, compaction, queueing, or dense
   fallback latency. It cannot establish throughput or a useful operating
   point.
4. The selector uses only previous accepted prefix, which is a deliberately
   weak state feature. It does not prove that a learned confidence policy is
   needed or worthwhile.

## Decision

Keep state-conditioned selection as an open gate rather than a result. The
next experiment must compare threshold policies and static uniform under
longer generation and held-out prompt splits, then feed the observed schedule
occupancy into the measured dense/grouped cost gate. The policy survives only
if its acceptance benefit remains after controller and fallback costs.

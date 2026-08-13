# Primary-Idea Decision After Held-Out Experiments

Date: 2026-08-14

## Decision

The primary architecture should be **prefix-survival-aware MLP scheduling**,
not static layer-width specialization.

The motivation still comes from unequal block/layer importance, but the most
defensible hardware primitive is a position/depth schedule that keeps dense
bidirectional attention and gates only position-wise MLP updates. The current
protected8 staircase schedule is:

```text
(5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1)
```

It reduces nominal MLP rows from 80 to 60 per DFlash block.

## Why this beats the layer-width story

| Candidate | Original calibration | Held-out screen | Decision |
|---|---|---|---|
| Layer 2 at 50% width | low-regret | mean prefix `0.9624` vs `1.0234` uniform | motivation / candidate only |
| Layer 3 at 50% width | not competitive | mean prefix `0.8696` | reject as default |
| Layers 2+3 at 50% width | joint loss | mean prefix `0.8561` | reject independent composition |
| Protected8 staircase | mean `1.2615` vs `1.2586` | 96-token mean `1.4448` vs `1.4219` | primary candidate |

The position schedule is still an acceptance experiment rather than a timing
result, but it is the only candidate that currently preserves acceptance in
both prompt sets and at both short/long budgets while providing a regular
hardware dataflow.

## Acceptance / work frontier

Using the measured draft-stage MLP fraction of 38%, the dense-attention plus
gated-MLP work proxy is:

```text
total_work_fraction = 0.62 + 0.38 * (scheduled_MLP_rows / 80)
```

For the held-out set:

| Schedule | MLP work fraction | Total draft-work proxy | Mean prefix | Mean / total-work proxy |
|---|---:|---:|---:|---:|
| Uniform | 1.000 | 1.000 | 1.0234 | 1.0234 |
| Protected8 conservative | 0.8375 | 0.9383 | 1.0078 | 1.0741 |
| Protected8 staircase | 0.7500 | 0.9050 | 1.4448 | 1.5965 |
| Protected4 moderate | 0.6375 | 0.8622 | 1.0312 | 1.1959 |

The staircase therefore improves the normalized committed-prefix-per-draft-
work proxy by approximately 12.3% on the 96-token screening set. This is not an
end-to-end throughput number: it excludes compaction, launch, queueing, and
verification latency. Protected4 is not promoted because it loses on the
original calibration workload despite looking good in this small screen.

## Final architecture story

**Motivation:** DFlash computes all block positions uniformly although only a
prefix is normally committed.

**Challenge:** bidirectional attention means low-value tail positions cannot be
naively deleted; their states can influence prefix states.

**Insight:** keep attention dense, but allocate MLP depth by a jointly
validated prefix-survival schedule.

**Hardware:** schedule descriptors, dense attention lanes, row-selective MLP
lanes, cross-request grouping, an occupancy estimator, and dense fallback.

**Chiplet role:** optional physical organization for the heterogeneous MLP
lanes. It is not required for the paper and remains a conditional result.

## Required claims and kill criteria

The paper may claim an acceptance-compatible dataflow and a calibrated
committed-value/work frontier. It may claim speedup only after a
correctness-tested grouped implementation beats dense fallback in the stated
batch regime.

Kill the heterogeneous execution claim if the protected8 candidate fails a
larger held-out workload or if compaction/fallback overhead removes the work
benefit. Do not revive static layer ranking or the simple previous-prefix
selector to rescue it.

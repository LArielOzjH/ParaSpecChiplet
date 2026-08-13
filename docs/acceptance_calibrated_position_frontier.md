# Acceptance-Calibrated Position-Schedule Frontier

Date: 2026-08-14

This note combines the 96-token held-out acceptance trace with the measured
RTX 4090 batch-64 row-latency calibration. It is the strongest current
hardware-facing result, but it remains a calibrated draft-stage proxy rather
than an end-to-end serving measurement.

## Inputs

Acceptance on the held-out 8-prompt, 96-token run:

| Schedule | Mean accepted prefix | `S1` | MLP work |
|---|---:|---:|---:|
| Uniform | 1.4219 | 0.6375 | 80 rows/layer-stack |
| Protected8 staircase | 1.4448 | 0.6530 | 60 rows/layer-stack |

Measured RTX 4090 row policy at batch 64:

- effective active rows: `[16, 16, 12, 10, 8]`;
- dense fallback for the first two layers;
- grouped execution for the last three layers;
- MLP latency: `5312.085 ms` uniform versus an estimated `4595.3 ms`
  staircase, a `13.49%` MLP-only reduction;
- batch 1/8 do not show a grouped payback and are outside the positive regime.

The raw calibration is in
[`data/gpu_mlp_row_sweep_batch_sizes_rtx4090.json`](../data/gpu_mlp_row_sweep_batch_sizes_rtx4090.json),
and the composed schedule table is in
[`data/calibrated_schedule_mlp_latency.json`](../data/calibrated_schedule_mlp_latency.json).

## Composed proxy

Let `f_MLP = 0.38` be the measured fraction of draft-stage time spent in MLP
and let `r_MLP = 0.1349` be the measured MLP reduction. The draft-stage proxy
is:

```text
draft_time_ratio = 1 - f_MLP * r_MLP = 0.9487
```

Combining acceptance and this time ratio gives:

```text
committed_prefix_per_draft_time_ratio
  = (1.4448 / 1.4219) / 0.9487
  = 1.0710
```

Thus the current best statement is a **7.1% acceptance-calibrated
draft-stage proxy improvement in the batch-64 regime**. This calculation does
not include attention timing, target verification, queueing, compaction launch
cost beyond the row calibration, or end-to-end service scheduling.

## Hardware interpretation

The result supports a concrete controller rule:

1. keep attention dense;
2. use the protected8 staircase row mask;
3. apply dense fallback whenever a layer's active group is not represented by
   a measured profitable row count;
4. restrict the positive claim to throughput serving with sufficiently large
   cross-request batches;
5. compare against equal-resource grouped monolithic execution before adding
   chiplet links.

The remaining decisive artifact is an integrated correctness-tested grouped
implementation. If its launch/queue overhead removes the measured 13.49% MLP
reduction, retain the acceptance/dataflow result but remove the speedup proxy.

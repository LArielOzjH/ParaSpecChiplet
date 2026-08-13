# Reconfigurable Fidelity-Fabric Cost Sweep

Date: 2026-08-14

This is a trace-independent analytical sweep of the execution substrate after
the held-out experiments showed that no single reduced-width block schedule is
universally safe. It asks when a fabric supporting uniform and protected8
staircase schedules should use grouped execution rather than dense fallback.

## Model

- block size: 16
- draft depth: 5
- uniform depth vector: `(5, ..., 5)`
- staircase depth vector: `(5,5,5,5,5,5,5,5,4,4,3,3,2,2,1,1)`
- normalized MLP work: one cycle per active row/layer
- schedule-launch cost: 10 cycles
- scatter cost: 0.1 cycles per active row
- grouped cost: active-row compute plus launch/scatter overhead
- dense cost: all rows at maximum depth plus dense launches

The numbers are normalized and are not an end-to-end timing claim. They are a
policy/fabric crossover screen; the RTX 4090 row measurements are the source
for later calibration.

## Arrival-order mixture

The table reports grouped cost divided by dense cost. A value above one means
the hardware should take the dense fallback. The request batch is filled with
the stated fraction of staircase schedules and the remainder uniform.

| Batch capacity | Staircase fraction | Grouped / dense |
|---:|---:|---:|
| 8 | 25% | 1.029 |
| 8 | 50% | 0.965 |
| 8 | 75% | 0.901 |
| 8 | 100% | 0.838 |
| 16 | 25% | 1.030 |
| 16 | 50% | 0.964 |
| 16 | 75% | 0.898 |
| 16 | 100% | 0.832 |
| 64 | 25% | 1.031 |
| 64 | 50% | 0.963 |
| 64 | 75% | 0.895 |
| 64 | 100% | 0.827 |

The crossover is therefore near a 40--50% staircase occupancy under these
assumptions. Grouping all requests unconditionally is wrong: a small minority
of heterogeneous schedules does not pay back its compaction overhead.

## Architecture implication

The minimum useful hardware mechanism is not a sparse MLP call. It is a
**reconfigurable schedule fabric** with:

1. a finite schedule descriptor and mode-specific MLP lanes;
2. an occupancy estimator that predicts grouped versus dense cost;
3. a short coalescing window for compatible schedules;
4. a dense fallback when the predicted occupancy is below the crossover;
5. equal-resource monolithic execution as the primary baseline.

Chiplets are optional. A chiplet split must additionally pay activation-link
and synchronization costs, so it is only credible if physical specialization
improves utilization or area/energy sharing enough to beat the grouped
monolithic fabric. The current width-aware chiplet sweep does not establish
that condition.

## What this does and does not show

This sweep shows a concrete architecture control rule and a workload regime
where heterogeneous execution can be useful. It does not show that a schedule
is acceptance-safe, that the selector predicts acceptance, or that the
normalized ratios equal GPU speedups. Those claims require the held-out
acceptance frontier, measured queue occupancy, and a correctness-tested fused
implementation.

The reproducible sweep entry point is
[`scripts/sweep_reconfigurable_fabric.py`](../scripts/sweep_reconfigurable_fabric.py),
with generated rows in
[`data/reconfigurable_fabric_sweep.json`](../data/reconfigurable_fabric_sweep.json).

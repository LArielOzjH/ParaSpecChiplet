# Acceptance-Calibrated Fidelity Scheduler

The first schedule selector combines official acceptance traces with the RTX
4090 reduced-width MLP measurements. It retains only schedules above an
accepted-prefix threshold and removes dominated schedules.

Using `min_accepted_prefix=1.4`, the measured frontier is:

| Schedule | Mean accepted prefix | Width vector | Calibrated MLP latency |
|---|---:|---|---:|
| layer 2 at 50% width | 1.4412 | `[1, 1, .5, 1, 1]` | 4,671 μs |
| uniform | 1.4525 | `[1, 1, 1, 1, 1]` | 5,182 μs |

The selected schedule saves about 9.9% of the calibrated MLP work while
remaining above the current acceptance threshold. This is not an
end-to-end speedup: the latency is a single-layer, batch-64 MLP calibration
and attention remains unmeasured in this selector. With the official stage
profile's roughly 38% MLP fraction, the corresponding conservative draft-stage
bound would be around 3.7%, before queueing and scheduling overhead.

The key architecture object is therefore a finite schedule table, not an
independent layer ranking:

1. collect joint acceptance for candidate width vectors;
2. reject schedules below the registered prefix-survival threshold;
3. use measured width/row latency to remove dominated choices;
4. group requests with the same schedule and fall back to dense execution if
   grouping overhead exceeds the calibrated saving.

The current selector is implemented by
`paraspec/fidelity_frontier.py` and
`scripts/analyze_fidelity_frontier.py`. Raw output is
`data/official_fidelity_frontier.json`.

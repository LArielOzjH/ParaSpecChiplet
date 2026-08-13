# DAC'27 Recommendation: What to Build and What to Kill

## Recommended paper

**SAGE-DFlash: Acceptance-Calibrated Reconfigurable Block-Fidelity Fabric for
Block-Parallel Speculative Decoding**

### Motivation

DFlash computes a block in parallel, but verification only commits a prefix.
Uniform draft computation therefore spends the same MLP work on states with
different committed-prefix value. Official probes confirm unequal block/layer
value, while held-out probes show that the ranking depends on workload or
entering state.

### Architectural insight

The right abstraction is not “always skip layer 2.” It is a finite,
reconfigurable schedule fabric:

```text
target hidden features
          |
   dense DFlash attention/backbone
          |
  schedule descriptor + occupancy gate
       /                 \
 full-width MLP       reduced-width MLP
       \                 /
        grouped execution / dense fallback
                    |
             target verification
```

Attention stays dense because block positions are bidirectionally coupled.
Only position-wise MLP fidelity changes. Compatible requests are grouped, but
the fabric falls back to dense execution when occupancy is below the measured
crossover.

## Three ideas and their roles

| Idea | Role | Motivation | Main challenge | Status |
|---|---|---|---|---|
| Reconfigurable grouped monolithic fabric | Primary | heterogeneous schedules can save MLP work when groups are sufficiently homogeneous | acceptance-safe selection plus compaction/fallback overhead | strongest surviving idea; hardware gate open |
| Static finite schedule table | Conservative baseline | removes controller overhead and isolates datapath value | held-out width screen does not yet show a universally safe vector | baseline/fallback, not headline |
| Heterogeneous chiplet fabric | Optional extension | physical specialization may improve lane utilization or sharing | link traffic and synchronization versus equal-resource monolithic | currently negative/conditional |

The selector must be reported in three forms: static uniform, offline oracle,
and state/workload-conditioned runtime selection. This prevents policy quality
from being confused with datapath quality.

## Evidence boundary

Supported:

- block/layer value is non-uniform;
- individually safe reductions do not compose independently;
- protected dense-attention MLP gating can preserve acceptance on a calibration
  workload;
- held-out results show that a fixed reduced-width layer ranking is not stable;
- grouped execution has an occupancy crossover and requires dense fallback;
- equal-resource chiplet models lose under the current traffic assumptions.

Not supported:

- a universally safe static layer schedule;
- generalized state-conditioned acceptance improvement (the first short probe
  is only preliminary);
- end-to-end throughput improvement;
- low-batch benefit;
- correctness-tested fused reduced-width kernel;
- chiplet advantage.

## Required decisive experiments

1. Build a held-out selector using previous accepted-prefix/state features and
   compare it against static uniform, static candidate schedules, and an
   offline oracle. The first threshold probe is recorded in
   `docs/state_conditioned_width_results.md`; extend it with longer generation,
   more prompt splits, controller overhead, and dense fallback.
2. Measure a trace-derived mixture of schedule classes in the queue model and
   validate the 40--50% occupancy crossover with real CUDA execution.
3. Only if the first two gates pass, implement a fixed-shape tensor-core-aware
   reduced-width kernel at batch 1/8/64 and test numerical equivalence.
4. Compare grouped monolithic and chiplet mappings with equal resources,
   activation bytes, synchronization, and queue fill. Remove chiplets from
   the headline if they lose.

## Kill criteria

- If state/workload selection cannot beat static uniform plus dense fallback,
  keep the schedule descriptor and datapath only as an oracle architecture
  study; do not claim adaptive policy value.
- If no candidate schedule survives held-out acceptance, kill fidelity
  reduction and retain only the DFlash profiling/occupancy study.
- If grouped execution does not beat dense in its target occupancy regime,
  kill the hardware speedup claim.
- If chiplet traffic loses to equal-resource monolithic, remove chiplets from
  the title and use the monolithic fabric as the paper architecture.

This framing gives the paper a single architecture story while allowing every
uncertain component to fail independently without reviving grammar-aware,
sparse-head, or chiplet-first ideas.

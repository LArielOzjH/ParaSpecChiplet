# DAC'27 Recommendation: What to Build and What to Kill

## Recommended paper

**SAGE-DFlash: Prefix-Survival-Aware MLP Scheduling for Block-Parallel
Speculative Decoding**

### Motivation

DFlash computes a block in parallel, but verification only commits a prefix.
Uniform draft computation therefore spends the same MLP work on states with
different committed-prefix value. Official probes confirm unequal block/layer
value, while held-out probes show that static layer-width ranking depends on
workload or entering state. The more stable candidate is a position/depth
schedule that preserves dense attention.

### Architectural insight

The right abstraction is not “always reduce layer 2.” It is a finite,
reconfigurable prefix-survival schedule fabric:

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
| Prefix-survival grouped monolithic fabric | Primary | position/depth schedules save MLP work while preserving dense context | acceptance-safe schedule plus compaction/fallback overhead | strongest surviving idea; hardware gate open |
| Static finite position/depth table | Conservative baseline | removes controller overhead and isolates datapath value | larger held-out acceptance and workload coverage | baseline/fallback, not headline |
| Heterogeneous chiplet fabric | Optional extension | physical specialization may improve lane utilization or sharing | link traffic and synchronization versus equal-resource monolithic | conditional: must exceed the measured 0.5--2.1% break-even gain |

The selector must be reported in three forms: static uniform, offline oracle,
and state/workload-conditioned runtime selection. This prevents policy quality
from being confused with datapath quality.

## Evidence boundary

Supported:

- block/layer value is non-uniform;
- individually safe reductions do not compose independently;
- protected dense-attention MLP gating can preserve acceptance on a calibration
  workload;
- the protected8 position/depth staircase preserves the acceptance summary in
  both the original and current held-out screening sets while removing 25% of
  nominal MLP rows;
- held-out results show that a fixed reduced-width layer ranking is not stable;
- grouped execution has an occupancy crossover and requires dense fallback;
- equal-resource chiplet models lose under the current traffic assumptions.

Not supported:

- a universally safe static layer schedule;
- generalized state-conditioned acceptance improvement; the 96-token follow-up
  rejects the simple previous-prefix threshold;
- end-to-end throughput improvement;
- low-batch benefit;
- correctness-tested fused reduced-width kernel;
- chiplet advantage.

## Required decisive experiments

1. Treat the previous-prefix threshold as a negative selector result. A richer
   state policy is optional future work, not a current claim; if revisited, it
   must compare against static uniform, static candidate schedules, and an
   offline oracle with controller overhead and dense fallback.
2. Measure a trace-derived mixture of schedule classes in the queue model and
   validate the 40--50% occupancy crossover with real CUDA execution.
3. Only if the first two gates pass, implement a fixed-shape tensor-core-aware
   reduced-width kernel at batch 1/8/64 and test numerical equivalence.
4. Compare grouped monolithic and chiplet mappings with equal resources,
   activation bytes, synchronization, and queue fill. Promote chiplets only if
   physical specialization exceeds the break-even boundary in
   `docs/chiplet_break_even_boundary.md`; otherwise remove chiplets from the
   headline.

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

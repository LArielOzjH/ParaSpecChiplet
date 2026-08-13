# DAC'27 Research Brief: Survival-Aware DFlash Execution

## One-sentence thesis

DFlash treats every speculative block position and every decode state as if it
had the same execution value, but verification commits a prefix and the value
of a new block depends on the state entering it; a hardware scheduler can use
that survival structure to select a protected-prefix, heterogeneous draft
schedule and map it onto shared and specialized compute resources.

The paper should lead with the execution mismatch, not with chiplets. Chiplets
are the physical organization that may make the required heterogeneity
efficient. A state-aware policy alone is not sufficient novelty: recent work
already explores dynamic speculative length, memory-aware draft selection, and
uncertainty repair. The architectural contribution must survive even with an
oracle schedule selector.

## Primary proposal: SAGE-DFlash

Working name: **State-Aware Grouped Execution (SAGE-DFlash)**.

For a block with accepted-prefix random variable `A`, define position value as
`S_i = P(A >= i)`. For a block entering state `q`, use `S_i(q)` and expected
committed value `E[A | q]`. The scheduler chooses a candidate depth vector
`d(q) = (d_1, ..., d_B)` and fidelity class subject to:

1. the first `K` positions remain protected at full fidelity;
2. tail positions may receive fewer upper draft layers or reduced precision;
3. shared lower layers are evaluated once before position-specific routing;
4. target verification remains authoritative and therefore preserves the
   speculative-decoding correctness contract;
5. all choices include activation movement, router, synchronization, and queue
   costs.

The hardware organization is:

```text
target hidden features
          |
   shared backbone chiplet  -- multicast --  prefix/full-fidelity chiplet
          |                                  \
          +-------------------------------> tail/cheap chiplet
                                             |
                                  scheduler + router + verifier interface
```

The router is not a generic network-on-chip addition: it carries the
layer-position survival map and state-selected schedule. A monolithic
accelerator with equal total resources is the required baseline.

## Why this is a DAC architecture problem

The novelty boundary is the interaction of three constraints:

- DFlash is block-parallel, but commitment is prefix-serial;
- DFlash attention is bidirectional within the block, so tail computation can
  affect protected prefix states;
- target hidden features are injected across draft layers, making layer
  specialization and activation reuse first-order dataflow concerns.

This is not a claim that positional loss weighting, dynamic draft length,
candidate pruning, early exit, quantization, or chiplets are individually new.
The contribution must be the survival-constrained dataflow and its
traffic-aware hardware realization.

## Evidence status

What is currently supported:

- A real Qwen3-0.6B/DFlash activation profile shows strong within-block
  correlation and therefore shared-computation potential.
- A four-prompt tail perturbation sweep shows that early perturbation and a
  short protected prefix are risky, while later perturbation and a longer
  protected prefix are much safer in that setup.
- An official DFlash adapter exists for lossless acceptance traces using
  `z-lab/dflash`; it correctly removes the target fallback token from the
  upstream statistic.
- The repository has state-conditioned survival, schedule-frontier, runtime
  controller, and chiplet-cost oracles with explicit tests.

What is not yet supported:

- no token acceptance result from the community 0.6B checkpoint, because its
  vocabulary mappings are invalid/missing;
- no selective-depth DFlash kernel or GPU timing result;
- no proof that state-conditioned schedules outperform a static schedule;
- no proof that chiplet traffic beats an equal-resource monolithic design.

These are open gates, not implied results.

## Three paperable variants and decision rule

| Variant | Core mechanism | Best case | Kill condition | Role |
|---|---|---|---|---|
| SAGE-DFlash | protected-prefix heterogeneous upper layers + physical shared/multicast dataflow; state conditioning is an input | schedule saves real draft work while preserving early acceptance | no layer-position asymmetry or no hardware benefit over monolithic design | primary |
| Protected-prefix DFlash | static shared-lower-layer plus safe prefix/tail boundary | interference is stable but state variation is weak | no stable boundary across prompts/models | fallback |
| Elastic DFlash fabric | chiplet/monolithic fabric with multicast and mixed-depth queues | heterogeneous requests cause substantial utilization loss on monolithic hardware | links, sync, or queueing erase compute savings | hardware extension |

The project should not present all three as equal contributions. Use the first
variant if the state-conditioned acceptance gate passes; use the second if
state variation is weak but layer-position asymmetry is stable; use only the
third as the lead if the hardware study independently demonstrates a strong
mixed-workload utilization problem.

## Required experiment sequence

### Gate 1: real trace and state variation

Run official DFlash on at least three workload families, multiple prompts, and
several requests per prompt. Report block size, draft depth, accepted prefix,
draft/verify latency, and request id. Measure:

- global `S_i` and hazard;
- `S_i(q)` for previous-prefix and confidence/context buckets;
- between-state variance and held-out calibration error;
- state transition counts and phase-change recovery.

If state-conditioned distributions are indistinguishable within confidence
intervals, kill the adaptive controller and retain the static fallback.

### Gate 2: selective-depth correctness

For the frontier candidates, run actual selective upper-layer execution with
the same prompts and target verifier. Compare vanilla uniform depth, uniformly
shallow depth, static staircase, state-aware schedule, and an ideal zero-router
oracle. Reject any candidate that violates the pre-registered protected-prefix
acceptance tolerance.

Activation cosine/L2 is only a screening signal; acceptance is the decision
metric.

### Gate 3: architecture and chiplet cost

Use measured per-layer MACs, activation bytes, and synchronization points. Sweep
multicast reuse, link bandwidth, router cycles, batch size, context length, and
mixed state/depth request distributions. Report:

- committed tokens per second;
- draft and verify latency separately;
- energy/area proxy or cycle-normalized cost;
- activation/link bytes;
- queueing and synchronization overhead;
- comparison with equal-resource monolithic execution.

If the chiplet implementation loses after traffic and synchronization, retain
the dataflow result as a monolithic architecture and explicitly kill the
chiplet claim.

## Recommended paper framing

**Motivation:** parallel drafting does not imply uniform speculative value.

**Challenge:** prefix commitment and bidirectional block attention conflict
with naive tail skipping.

**Insight:** protect an empirically safe prefix boundary, share lower-layer
context, and spend upper-layer fidelity according to state-conditioned survival
value.

**Architecture:** a survival-map scheduler, shared backbone, heterogeneous
prefix/tail execution, and traffic-aware routing.

**Novelty discipline:** LibraSpec, CURE, MemSpec, and xPress are close
algorithm/runtime neighbors. The paper must therefore include a static and an
oracle-policy experiment on the same physical fabric; policy selection alone
cannot carry the contribution.

**Evaluation claim:** only claim an improvement if acceptance, latency, traffic,
and equal-resource baselines all support it.

## Current recommendation

Continue with SAGE-DFlash as the primary research hypothesis, but do not yet
commit the paper to chiplets or adaptive scheduling. The next decisive artifact
is an official multi-request acceptance trace. Until that exists, the strongest
defensible statement is that the activation evidence motivates a protected
prefix and that the repository has a falsifiable path to test whether it becomes
a real architecture opportunity.

# Draft-Block Importance: Candidate DAC'27 Direction

## Status

This is a research hypothesis under validation, not an established result. The
working interpretation of “block” is a Transformer block/layer inside the
DFlash draft model. Grammar-aware decoding and sparse-head execution are out
of scope.

## Hypothesis

DFlash does not require every draft Transformer block to receive the same
amount of execution. A block's value is its marginal contribution to the
accepted prefix, conditioned on the block state and workload phase. Define

\[
  V_l(q) = \frac{\mathbb{E}[A\mid q,\text{full}] -
                 \mathbb{E}[A\mid q,\text{bypass }l]}
                {C_l + M_l + Q_l},
\]

where `A` is the accepted prefix length, `q` is the state entering the DFlash
cycle, and the denominator includes compute, movement, and queueing cost. The
architecture should allocate fidelity and physical resources according to
`V_l(q)`, rather than replicate all draft blocks uniformly.

This is distinct from token pruning: the unit of scheduling is a draft
Transformer block, while target verification remains authoritative.

## Candidate architecture

The safest staged design is **dense-attention, block-heterogeneous execution**:

1. Keep each block's bidirectional attention dense while the hypothesis is
   being tested; this preserves cross-position context.
2. Measure block importance using full-block bypass and MLP-only bypass.
3. Map high-value blocks to full-fidelity lanes and low-value blocks to
   reduced-precision or reduced-MLP lanes.
4. Use a shared lower-layer data path and specialized upper-layer chiplets
   only when activation movement and synchronization cost are lower than the
   saved work.
5. Fall back to an equal-resource monolithic schedule when the batch is too
   small or block queues are heterogeneous.

Full layer bypass is therefore an analysis primitive first, not automatically
the implementation mechanism.

## Alternatives and recommendation

### Static block specialization

Use one offline importance ranking for all requests. It is easiest to build and
to compare against an equal-resource monolithic accelerator, but it fails if
layer importance varies strongly by prompt or decode phase.

### State-conditioned block scheduling

Select a precompiled layer/fidelity schedule from a small state table using
recent accepted-prefix history. It can exploit phase variation, but controller
overhead and policy novelty are risks; it should remain an extension until a
static ranking is shown to be insufficient.

### Elastic block fabric

Use shared lower layers plus heterogeneous upper-layer chiplets and queues.
This gives the strongest architecture story, but only if a traffic-aware,
equal-resource comparison shows a real utilization advantage. Chiplets are a
physical realization, not the motivation by themselves.

Recommended order: validate static block importance, then test state
conditioning, and only then make the chiplet fabric part of the headline.

## Decisive experiment matrix

Run the official Qwen3-4B/DFlash-b16 serving loop with the same prompts and
target verifier:

| Experiment | Purpose | Metric |
|---|---|---|
| Full 5-layer baseline | Reference | `S1/S2/S4`, accepted prefix, latency |
| Bypass each layer `l` | Estimate block marginal value | acceptance drop by layer |
| Truncate to 1/2/3/4 layers | Test cumulative depth | acceptance versus work |
| Bypass only MLP of `l` | Separate attention and MLP value | acceptance and stage time |
| Prompt/phase split | Test stability of ranking | held-out ranking correlation |
| Mixed batch schedules | Test serving feasibility | queueing, utilization, latency |

The committed-token metric is primary. Hidden-state cosine/L2 is only a
screening signal. Python hooks and layer replacement are acceptance probes,
not speedup measurements.

## Architecture evaluation gate

For every candidate schedule, compare:

- vanilla uniform DFlash;
- uniform reduced-depth DFlash;
- static block-specialized monolithic execution;
- state-conditioned execution;
- equal-resource chiplet execution;
- an ideal zero-routing oracle.

The model must include per-layer MACs, activation bytes, multicast reuse,
router cost, synchronization, queue fill, and dense fallback. Report committed
tokens per second, draft and verify latency separately, activation/link bytes,
and area/energy-normalized cost where available.

## Kill conditions

Kill or demote the hypothesis if any of the following holds:

- layer rankings are unstable across held-out prompts or phases;
- bypassing every layer causes a similar acceptance loss;
- MLP-only approximation does not preserve early-prefix acceptance;
- heterogeneous queues lose to an equal-resource monolithic engine after
  movement and synchronization costs;
- the state controller matches the static schedule after overhead;
- gains exist only in a proxy and disappear in the official serving loop.

## Provisional DAC framing

**Motivation:** block-parallel speculation exposes a new mismatch between
uniform draft-layer execution and non-uniform contribution to committed
prefixes.

**Challenge:** DFlash's bidirectional attention couples positions across the
block, and layer specialization introduces activation traffic and queueing
overhead.

**Potential contribution:** a survival/value-aware block dataflow with
traffic-aware heterogeneous execution, evaluated against an equal-resource
monolithic baseline.

The current repository probe for the first experiment is
`scripts/probe_official_block_ablation.py`. Until its official results exist,
the existing protected-prefix grouped MLP work remains the defensible fallback
paper core.

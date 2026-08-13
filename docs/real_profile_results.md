# Real local activation profile

Date: 2026-08-14

## Setup

- Target: `Qwen/Qwen3-0.6B`
- Draft: `Eros483/Qwen3-0.6B-DFlash-shift-b8`
- Draft layers: 3
- Block size: 8
- Context length: 126 tokens
- Random anchors: 8
- Execution: CPU, Transformers 5.12.1, Speculators 0.6.0.1

The run used the target's auxiliary hidden states and captured each DFlash draft-layer output with forward hooks. It measured representation geometry only.

## Result

| Draft layer | Mean off-diagonal cosine | Mean adjacent cosine | Position-1 norm | Position-8 norm |
|---:|---:|---:|---:|---:|
| 1 | 0.8139 | 0.9139 | 140.09 | 102.68 |
| 2 | 0.8139 | 0.9166 | 238.26 | 187.59 |
| 3 | 0.8986 | 0.9537 | 333.87 | 318.84 |

The block positions are strongly correlated, especially adjacent positions. This is evidence for a shared-computation opportunity. It is not evidence that tail positions can be skipped safely: bidirectional attention may make tail states useful to prefix states.

## Tail perturbation result

As a first coupling probe, positions 5–8 were replaced after a chosen draft layer by their per-block mean. The final prefix states changed very little:

| Perturbation point | Mean prefix cosine | Mean prefix relative L2 |
|---:|---:|---:|
| after layer 1 | 0.99999994 | 0.0001662 |
| after layer 2 | 1.00000000 | 0.0000954 |

This is encouraging evidence that a protected prefix may tolerate a low-fidelity tail in this setup. It is not a general result: the experiment uses one checkpoint, one prompt template, eight anchors, and a mean replacement much milder than skipping all tail computation. The next sweep must vary the replacement, protected-prefix boundary, prompts, and layer count.

A stronger perturbation check replaced tail states by a copy of position 4 or by zero. Copying position 4 produced effectively zero prefix relative-L2 change. Zeroing the tail after layer 1 produced 0.37% mean prefix relative-L2 change (cosine 0.999868); zeroing after layer 2 again produced a negligible change in this setup. This suggests the first draft layer is the only plausible coupling boundary here, but the result is still a single-model/single-prompt observation and must not be generalized without a multi-workload sweep.

## Critical caveat

The checkpoint declares a 32,000-token draft vocabulary and a 151,936-token target vocabulary but does not ship `t2d/d2t` mappings. Therefore no token-level acceptance result is reported from this run. The local token-agreement experiment produced invalid conclusions until that mapping is supplied. The activation result remains useful because it does not depend on vocabulary IDs.

## Next experiment

Use a checkpoint with valid vocabulary mappings or run the official serving path. Then add a tail perturbation test:

1. run the full block;
2. replace tail-position states after a chosen layer with a shared mean state or stop their upper-layer updates;
3. measure the change in prefix hidden states and target acceptance;
4. sweep the protected prefix boundary.

The primary idea survives only if a non-trivial tail saving exists while prefix acceptance remains stable.

## Multi-condition tail perturbation sweep

Date: 2026-08-14

The follow-up sweep used the same target/draft pair and four prompts. It varied
the protected prefix (`1, 2, 4, 6`), replaced the remaining tail with its mean,
copied the preceding position, or zeroed it, and applied the perturbation after
draft layer 1 or 2. The raw output is
`data/local_tail_perturbation_sweep_v2.json`.

The aggregate prefix-state results were:

| After layer | Mode | Protected prefix | Mean cosine | Mean relative L2 |
|---:|---|---:|---:|---:|
| 1 | copy previous | 1 | 0.995830 | 5.49% |
| 1 | copy previous | 2 | 0.999243 | 1.82% |
| 1 | copy previous | 4 | 0.999985 | 0.24% |
| 1 | mean | 1 | 0.999886 | 0.77% |
| 1 | mean | 2 | 0.999987 | 0.25% |
| 1 | mean | 4 | 0.999999 | 0.04% |
| 1 | zero | 1 | 0.998424 | 3.95% |
| 1 | zero | 2 | 0.999132 | 2.75% |
| 1 | zero | 4 | 0.999642 | 1.70% |
| 2 | zero | 4 | 0.999974 | 0.35% |

The omitted layer-2 mean and copy-previous cases are even closer to the
baseline; the complete table is in the JSON artifact. The pattern is stable
in direction across the four prompts: early perturbation and a short protected
prefix are the risky cases, while perturbation after layer 2 or protection of
the first four positions is much safer.

This supports a protected-prefix boundary rather than unconditional tail
skipping. It still does not establish token acceptance or end-to-end speedup:
the checkpoint has invalid vocabulary mappings, and the experiment perturbs
activation states rather than implementing a reduced-depth kernel. The next
gate is a valid acceptance trace plus a cost model for the safe boundary.

## Candidate valid-vocabulary checkpoint

Metadata inspection found
`orestis-z/dflash-qwen3-0.6b-microcycle-dflash`, whose configuration declares a
draft vocabulary of 151,936 entries, matching `Qwen/Qwen3-0.6B`. Its public
validation fixture is preserved at
`data/qwen3_0.6b_microcycle_dflash_val_metrics.json`; it reports decreasing
marginal position accuracy and EAL 1.0749. This is a better candidate for the
official verifier path than the earlier 32k-vocabulary checkpoint. The fixture
still does not contain cycle-level accepted prefixes, so it is evidence for
checkpoint compatibility and position asymmetry only, not an acceptance
result. The full 723 MB weight download was not completed in the current
low-resource network environment; the GPU capture script can use the model
directly on a machine with normal Hugging Face transfer.

The public marginal metrics alone permit only broad Fréchet bounds. More
importantly, `full_acc_epoch=0.288` exceeds the listed position-4 marginal
accuracy `0.213`, which is impossible if the fields describe the same event
and mask. The metrics therefore have incompatible definitions (or masks), and
the repository refuses to combine them into a claimed prefix-survival trace.
`paraspec.survival_bounds` retains the consistency check as a warning tool;
official cycle-level acceptance remains required.

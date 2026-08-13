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

## Offline agreement probe with target-sized vocabulary

The downloaded `orestis-z/dflash-qwen3-0.6b-microcycle-dflash` checkpoint was
loaded with the Speculators backend and evaluated on four prompts with eight
fixed random anchors per prompt. The probe resets the RNG before the draft
forward so the internally sampled anchors align with the target positions. Raw
events are in `data/local_offline_acceptance_proxy_orestis.json`.

| Accepted draft prefix | Number of anchors |
|---:|---:|
| 0 | 14 |
| 1 | 6 |
| 2 | 7 |
| 3 | 3 |
| 4 | 2 |

The mean is 1.156 accepted positions out of a block size of 8. This is useful
as a smoke test for target-sized vocabulary and prefix comparison, but it is
not an official serving result: the Speculators forward evaluates randomly
anchored masked blocks under a fixed context, not an autoregressive decode
loop. The checkpoint also lacks the optional `verifier_lm_head` used for its
training metrics; the probe ignores that loss output and only uses draft token
predictions. Official GPU serving remains the acceptance gate.

## Selective-depth offline proxy

The next probe implemented a protected-prefix depth schedule by replacing a
selected tail position's layer output with its layer input after the layer
forward. This preserves the rest of the bidirectional computation and measures
agreement degradation before a real selective kernel exists. The raw output is
`data/local_selective_depth_proxy_orestis.json`.

| Schedule | Depth vector | Layer-work sum | Mean accepted prefix |
|---|---|---:|---:|
| uniform | `(3,3,3,3,3,3,3,3)` | 24 | 1.156 |
| protected staircase | `(3,3,3,2,2,2,1,1)` | 17 | 1.094 |

The staircase reduces nominal per-position draft-layer work by 29.2% while
losing 5.4% of this offline proxy's mean accepted prefix. This is an
encouraging screening signal for a protected-prefix dataflow, not a serving
acceptance or speedup result: skipped layers are still physically executed in
the probe, and the blocks are masked training-style anchors rather than an
autoregressive decode loop. The candidate must be re-evaluated with official
verification and a real selective-depth implementation.

## Selective-depth design-space sweep

The same four prompts and fixed anchors were used to sweep five schedules. Raw
records are in `data/local_selective_depth_sweep_orestis.json`; `layer-work sum`
is the sum of per-position executed draft layers.

| Schedule | Depth vector | Layer-work sum | Mean accepted prefix | Change vs uniform |
|---|---|---:|---:|---:|
| uniform | `(3,3,3,3,3,3,3,3)` | 24 | 1.156 | — |
| protected4 conservative | `(3,3,3,3,3,2,2,1)` | 20 | 1.156 | 0.0% |
| protected4 staircase | `(3,3,3,3,2,2,1,1)` | 18 | 1.094 | -5.4% |
| protected2 staircase | `(3,3,3,2,2,2,1,1)` | 17 | 1.031 | -10.8% |
| aggressive tail | `(3,3,2,1,1,1,1,1)` | 13 | 0.625 | -45.9% |

This small sweep reveals a useful screening Pareto region: protecting the first
four positions permits a conservative 16.7% nominal work reduction with no
observed proxy loss, while more aggressive schedules degrade sharply. It is
not evidence of real speedup or serving acceptance; all skipped layers are
still executed in the proxy. The result motivates testing protected-prefix
boundaries 4 and 6 first on a GPU selective-depth implementation.

## Attention-preserving MLP-only proxy

The full-layer proxy above is conservative for context but overstates the
amount of work that a hardware design can save. A more targeted mechanism keeps
bidirectional attention for every block position, but zeros the residual MLP
update for tail positions after their scheduled depth. This preserves the
attention context path while removing the position-specific upper-layer MLP
work—the first microarchitecture candidate that can plausibly be implemented
without breaking block context.

The same four prompts and schedules were run with `--mode mlp`; raw records are
in `data/local_selective_mlp_proxy_orestis_v2.json`. In this v2 path, the hook
passes only non-skipped position rows through the MLP and scatters zeroes for
skipped rows. Thus the reported MLP work is an actual reduced-row CPU
execution path, rather than a full MLP followed by an output mask.

| Schedule | MLP-work sum | Mean accepted prefix | Change vs uniform |
|---|---:|---:|---:|
| uniform | 24 | 1.156 | — |
| protected4 conservative | 20 | 1.156 | 0.0% |
| protected4 staircase | 18 | 1.125 | -2.7% |
| protected2 staircase | 17 | 1.063 | -8.1% |
| aggressive tail | 13 | 0.750 | -35.1% |

The proxy suggests a stronger operating point than whole-layer skipping:
protected4 conservative saves 16.7% of MLP work with no observed agreement
loss, and protected4 staircase saves 25% with only a 2.7% loss. The numbers
are unchanged from the earlier screening run, but the execution path now
matches the intended row-selective mechanism. This is still not measured GPU
speedup or serving acceptance: attention is fully executed, and the probe uses
masked training-style blocks rather than an autoregressive decode loop. The
next hardware model should separate dense attention cost from selectively
gated MLP cost.

## Official GPU acceptance trace

Date: 2026-08-14

The official `z-lab/dflash` generation path was run on an NVIDIA RTX 4090
(49 GiB, CUDA 12.8) with the local `Qwen3-4B` target and
`Qwen3-4B-DFlash-b16` draft. The draft has five layers and block size 16. A
12-prompt sweep spanning systems explanations, inference concepts, coding,
debugging, and general technical questions produced 480 decode cycles. Raw
events are in `data/official_qwen3_4b_dflash_trace_v2.jsonl`; the prompt list is
`data/official_trace_prompts.txt`.

The accepted-prefix distribution was:

| Accepted prefix | Cycles |
|---:|---:|
| 0 | 187 |
| 1 | 103 |
| 2 | 60 |
| 3 | 37 |
| 4 | 12 |
| 5 | 6 |
| 6 | 7 |
| 7 | 7 |
| 8 | 1 |
| 9 | 1 |
| 10 | 1 |
| 13 | 1 |
| 15 | 1 |

The mean accepted prefix is 1.423 draft tokens. Prefix survival falls from
`S1 = 0.596` to `S2 = 0.346` and `S4 = 0.096`; no cycle accepted all 16 draft
positions. Per-prompt means range from 0.865 to 2.345, a 2.7x spread. This is
direct evidence that uniform execution value across a block is a poor model,
and that workload/entering-state conditioning is worth studying.

An exploratory descriptive state analysis, using the previous accepted prefix
as the entering-state key, is stored in
`data/official_qwen3_4b_state_analysis_b4.json`. It shows different estimated
expected committed values for coarse previous-prefix buckets, but it is not a
causal policy result: prompt and generation phase are confounded, and some
buckets are small. The trace contains official acceptance lengths and timing
metadata, but no confidence vectors, token IDs, or per-layer checkpoints.
Therefore it supports the survival-heterogeneity motivation, not yet a claim
that an adaptive schedule improves acceptance or end-to-end latency.

## Official attention-preserving MLP-gating acceptance

Date: 2026-08-14

The official Qwen3-4B/DFlash-b16 path was then rerun with a fixed schedule
installed directly into the draft model. Every draft layer kept dense
bidirectional attention, while only the non-skipped position rows entered the
MLP. The target verifier and autoregressive decode loop were unchanged. Raw
events are in `data/official_qwen3_4b_mlp_gating.jsonl`; the implementation is
`paraspec/official_selective.py` and
`scripts/probe_official_mlp_gating.py`.

| Schedule | MLP work | Mean accepted prefix | S1 | S2 | S4 |
|---|---:|---:|---:|---:|---:|
| uniform | 80 | 1.259 | 0.569 | 0.325 | 0.083 |
| protected8 conservative | 67 (-16.25%) | 1.259 | 0.575 | 0.330 | 0.080 |
| protected8 staircase | 60 (-25.00%) | 1.261 | 0.566 | 0.330 | 0.080 |
| protected4 moderate | 51 (-36.25%) | 1.220 | 0.559 | 0.331 | 0.079 |

This is the first official serving-loop evidence that a heterogeneous
upper-layer MLP schedule can preserve prefix acceptance while leaving the
bidirectional attention context intact. The protected8 staircase is the
current primary candidate: it removes 25% of nominal MLP rows with no material
change in the measured acceptance summary. The protected4 schedule is an
aggressive stress point with an approximately 3.1% mean-prefix reduction.

The measured end-to-end per-output-token timing did not improve: the current
Python implementation averaged roughly 10.08 ms/token for uniform,
10.67 ms/token for protected8 conservative, and 10.70 ms/token for protected8
staircase. This is expected because row gathering/scattering is not a fused
GPU kernel and can cost more than the saved MLP work. These numbers are not a
speedup claim; they establish an acceptance-preserving execution target for a
real fused selective-MLP kernel. Dense attention cost, gather/scatter cost,
and synchronization must be included in the eventual architecture model.

Using the published draft dimensions (`hidden=2560`, `intermediate=9728`,
Q/K/V/O projection widths `4096/1024/1024/4096`) as a first MAC screen gives
approximately 26.2M projection MACs for attention and 74.7M MACs for the
SwiGLU MLP per position and layer, before attention score/value terms. Under
these assumptions, the protected8 staircase's 25% MLP-row reduction becomes
an approximately 18.5% reduction in total draft projection work, because
dense attention remains paid for every position. This is the relevant
architecture-level opportunity; reporting the raw 25% MLP reduction as a
25% end-to-end speedup would be incorrect.

## GPU MLP row-selection microbenchmark

To separate matrix-multiplication effects from the end-to-end serving result,
the draft MLP from `Qwen3-4B-DFlash-b16` was benchmarked on the same RTX 4090
with 16 positions versus 9 active positions (the protected8 staircase's first
tail-layer shape). Results are one-layer CUDA timings after warmup:

| Request batch | Dense 16 rows (ms) | Active 9 rows (ms) | Gather + MLP + scatter (ms) |
|---:|---:|---:|---:|
| 1 | 0.182 | 0.180 | 0.260 |
| 8 | 0.216 | 0.213 | 0.311 |
| 64 | 1.062 | 0.675 | 0.795 |

The small-batch result is a warning: row selection does not automatically
produce useful GPU savings, and explicit gather/scatter can lose. At batch 64,
however, active-row execution reduces dense MLP latency by about 36.4%, or
about 25.2% after the measured gather/scatter path. This supports a concrete
architecture challenge: the design needs cross-request position grouping,
persistent/fused row compaction, or dedicated heterogeneous lanes to avoid
launch and movement overhead. The benchmark is a single-layer microbenchmark,
not an end-to-end speedup result.

A finer active-row sweep at batch 64 is preserved in
`data/gpu_mlp_row_sweep_rtx4090.json` and reproducible with
`scripts/benchmark_mlp_rows.py`. The dense MLP remains near 1.03 ms across
row counts, while grouped execution falls from 1.167 ms at 16 active rows to
0.817 ms at 9 rows and 0.406 ms at 4 rows. The 16-row grouped case is slower
than dense because compaction is pure overhead. This gives the queue model a
measurable operating curve: a fused implementation must avoid paying the
compaction tax when the selected schedule is effectively uniform.

Applying an exact-row dense fallback policy to the measured curve changes the
nominal savings. The protected8 staircase becomes 22.5% realizable MLP-row
savings because its 14/16 layer falls back to dense; protected8 conservative
becomes 13.75%. The protected4 schedule also becomes 22.5% under the
conservative rule because its 5/16 point was not directly calibrated and falls
back to dense. The full per-layer decisions are in
`data/official_row_policy_sweep.json` and the policy implementation is
`paraspec/row_policy.py`. This is a hardware execution estimate, not a new
acceptance run.

## Batch-aware dense fallback calibration

The row sweep was repeated at batch sizes 1, 8, and 64; the complete table is
in `data/gpu_mlp_row_sweep_batch_sizes_rtx4090.json`. At batch 1 and 8, every
tested active-row count was slower with grouped gather/scatter than dense MLP.
At batch 64, grouped execution became faster at 12/16 active rows and below.
This establishes that the fallback threshold is two-dimensional: active-row
count alone is insufficient; the controller must also observe effective batch
size. The calibrated policy API accepts `batch_size` and conservatively rejects
missing batch-specific measurements.

As a software-fusion sanity check, a fixed-shape `torch.compile` MLP wrapper
was tested at batch 64 and 9/16 active rows. It measured 1.130 ms with the
compiled MLP plus eager scatter, versus 0.823 ms for eager grouped execution
and 1.061 ms for dense. The result is preserved in
`data/gpu_compile_wrapper_rtx4090.json`. Generic compilation did not remove
the movement overhead; a positive low-batch result therefore requires a
purpose-built persistent/ fused lane implementation, not merely a compiler
flag.

Applying that policy to the four candidate schedules gives no realizable row
saving at batch 1 or 8; at batch 64, protected8 staircase retains 22.5% and
protected8 conservative 13.75%. The complete per-batch table is preserved in
`data/calibrated_schedule_batch_policy.json`. This makes the current claim
explicitly a throughput-serving claim, not a single-request latency claim.

## Mixed-schedule grouped execution

The grouping question was tested with a deliberately mixed batch: half of the
requests used all 16 rows and half used 9 active rows. The MLP output was
scattered back into the original batch layout in both selective variants.

| Request batch | Dense | One grouped active call + scatter | Separate schedule calls + scatter |
|---:|---:|---:|---:|
| 8 | 0.216 ms | 0.289 ms | 0.394 ms |
| 64 | 1.020 ms | 0.998 ms | 1.813 ms |

At batch 64, grouping avoids the severe penalty of separate schedule calls,
but the net gain over dense execution is only about 2.1% because 78.1% of rows
remain active. At batch 8, even grouped execution loses. This supports a
stronger queueing requirement: the scheduler must coalesce requests with
similar depth vectors, or the heterogeneous execution opportunity is diluted
by mixture overhead. A chiplet design must therefore model schedule queues,
not just aggregate MAC reduction.

The same state-to-schedule stress analysis is reproducible with
`scripts/analyze_trace_queue.py`; its illustrative output is preserved in
`data/official_trace_queue_sensitivity.json`. With the explicitly normalized
cost parameters in that file, coalescing reduces the mean number of schedule
groups from about 1.8 to 1.0, but creates one extra partially filled batch.
This is a queueing sensitivity result, not evidence that the proposed schedule
causes the observed acceptance distribution.

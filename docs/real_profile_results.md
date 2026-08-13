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

## Critical caveat

The checkpoint declares a 32,000-token draft vocabulary and a 151,936-token target vocabulary but does not ship `t2d/d2t` mappings. Therefore no token-level acceptance result is reported from this run. The local token-agreement experiment produced invalid conclusions until that mapping is supplied. The activation result remains useful because it does not depend on vocabulary IDs.

## Next experiment

Use a checkpoint with valid vocabulary mappings or run the official serving path. Then add a tail perturbation test:

1. run the full block;
2. replace tail-position states after a chosen layer with a shared mean state or stop their upper-layer updates;
3. measure the change in prefix hidden states and target acceptance;
4. sweep the protected prefix boundary.

The primary idea survives only if a non-trivial tail saving exists while prefix acceptance remains stable.


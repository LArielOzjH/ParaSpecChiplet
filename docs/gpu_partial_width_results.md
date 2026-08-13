# GPU Reduced-Width MLP Results

## Setup

- GPU: RTX 4090
- Draft MLP: Qwen3-4B-DFlash-b16, hidden size 2560
- Intermediate size: 9728
- Batch: 64 requests
- Operation: gated MLP with dense matrix multiplications on active rows

The benchmark truncates the intermediate dimension to 75%, 50%, or 25% and
uses the corresponding slices of gate, up, and down projection weights. It is
a reduced-compute proxy for partial MLP fidelity/reduced lanes. It does not
measure acceptance, energy, area, or end-to-end serving speedup.

## Selected measurements

| Active rows/request | Full-width (ms) | 75% (ms) | 50% (ms) | 25% (ms) |
|---:|---:|---:|---:|---:|
| 16 | 1,034 | 782 | 525 | 272 |
| 12 | 850 | 634 | 435 | 218 |
| 9 | 677 | 511 | 340 | 178 |
| 8 | 555 | 427 | 294 | 173 |
| 4 | 318 | 240 | 174 | 99 |
| 2 | 213 | 160 | 104 | 55 |

The full-width and reduced-width paths are both evaluated on the same active
rows. At batch 64, 50% width roughly halves MLP latency and 25% width reduces
latency by roughly 3--4x across the tested row counts. This is materially
different from Python row compaction: reduced width keeps the matrix shapes
regular and avoids gather/scatter as the primary optimization.

## Interpretation and limits

This supports a plausible hardware mechanism for the partial-fidelity
acceptance results: high-value blocks can retain full-width MLPs while lower-
value blocks use reduced-width lanes. The current implementation slices
pretrained weights and is not a trained low-width model. Therefore the next
experiment must run the same reduced-width MLP inside the official DFlash
serving loop and measure `S1/S2/S4`. Until that acceptance gate passes, these
numbers are only a compute-side result.

Raw measurements: `data/gpu_mlp_width_rows_rtx4090.json`.

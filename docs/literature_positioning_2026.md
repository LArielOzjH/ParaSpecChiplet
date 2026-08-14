# Literature Positioning Audit

Date: 2026-08-14

This is a focused positioning snapshot for the DAC'27 direction. It is not a
complete survey; it identifies the closest conceptual neighbors and the
novelty boundary SAGE-DFlash must defend.

## Direct DFlash and block-diffusion neighbors

| Work | What it contributes | Boundary for SAGE-DFlash |
|---|---|---|
| [DFlash](https://arxiv.org/abs/2602.06036) | Block-diffusion drafter that generates a block in one pass and uses target verification. | Defines the execution substrate; does not provide our architecture-level fidelity/occupancy dataflow. |
| [DFlare](https://arxiv.org/abs/2606.02091) | Scales draft capacity through richer layer-wise target-feature fusion. | Model/training capacity scaling; orthogonal to scheduling MLP work after the draft is built. |
| [DeLS-Spec](https://arxiv.org/abs/2607.07409) | Adds a decoupled short-context correction head to improve DFlash drafting. | Draft algorithm/training; does not address hardware execution of heterogeneous block fidelity. |
| [xPress](https://arxiv.org/abs/2608.02438) | Parallel causal refinement for diffusion-drafter outputs. | Restores causal dependence in the draft; our challenge is preserving dense context while reducing MLP work. |
| [DDTree](https://arxiv.org/abs/2604.12989) and [DominoTree](https://arxiv.org/abs/2607.08642) | Build and verify trees from block-diffusion or causally corrected drafts. | Changes candidate structure/verification; our unit is MLP fidelity inside one block and target verification remains authoritative. |

These works make a static “DFlash is parallel” motivation insufficient. The
paper must state the narrower systems mismatch: DFlash's positions are drafted
in parallel, but their target-verifier value is prefix-shaped and their MLP
work is not equally useful.

## Speculative-decoding systems and hardware-aware scheduling

| Work | Relevant lesson | SAGE-DFlash distinction |
|---|---|---|
| [Speculative Decoding](https://arxiv.org/abs/2203.16487) | Establishes the lossless draft/verify contract. | We preserve this contract and use accepted prefix as the hardware value metric. |
| [Decoding Speculative Decoding](https://arxiv.org/abs/2402.01528) | Shows draft latency and hardware efficiency can matter more than draft LM quality. | Supports our cost-aware schedule motivation; our granularity is DFlash position/depth MLP dataflow. |
| [Sequoia](https://arxiv.org/abs/2402.12374) | Hardware-aware tree optimization and platform-dependent speculation. | Tree budget optimization is a baseline/neighbor; SAGE selects fidelity schedules under dense block attention. |
| [The Synergy of Speculative Decoding and Batching](https://arxiv.org/abs/2310.18813) | Shows optimal speculation depends on batch size and motivates adaptive serving. | Reinforces our batch/occupancy evaluation; our adaptation is at MLP execution and schedule grouping, not speculation length alone. |
| [AMUSD](https://arxiv.org/abs/2410.17375) | Asynchronous multi-device draft/verify execution. | Multi-device overlap is a system-level neighbor; our chiplet option is conditional and must pay explicit activation traffic/sync. |
| [S3D](https://arxiv.org/abs/2405.20314) | Mid-layer skipping and simultaneous self-speculation for low-memory GPUs. | Layer skipping is a direct baseline; our evidence shows static block-width skipping is not stable and retains dense attention. |
| [EQSPEC/EXSPEC](https://arxiv.org/abs/2510.22876) | Batch correctness invariants and grouping of same-length sequences. | Motivates explicit queue/synchronization accounting; SAGE applies the invariant discipline to fidelity schedules. |
| [HADES](https://arxiv.org/abs/2412.19925) | Hardware-level speculative decoding accelerator concept. | Hardware SD alone is not the novelty; SAGE contributes a DFlash-specific acceptance/fidelity dataflow and negative cost boundaries. |

## Recent algorithmic neighbors that constrain novelty

Recent DFlash-adjacent papers also explore prefix-aware training ([Spec-AUF](https://arxiv.org/abs/2607.01893)), richer target-feature alignment ([DFlare](https://arxiv.org/abs/2606.02091)), and causal correction ([xPress](https://arxiv.org/abs/2608.02438)). Therefore the following are not sufficient contributions by themselves:

- a position-weighted loss;
- dynamic draft length;
- a causal refinement head;
- a better tree search;
- a generic early-exit or token-pruning controller;
- a chiplet partition without an equal-resource traffic comparison.

## Defensible novelty statement

SAGE-DFlash is an architecture/dataflow study of **prefix-survival-aware
fidelity scheduling inside a block-parallel diffusion drafter**. Its novelty
claim should be the conjunction of:

1. dense attention preserved under bidirectional block coupling;
2. jointly measured position/depth MLP schedules rather than independent layer
   ranking;
3. grouped execution with an occupancy-driven dense fallback;
4. explicit acceptance, queue, activation movement, synchronization, and
   equal-resource fabric accounting.

The chiplet implementation is a conditional result. A reviewer should be able
to remove chiplets from the paper without removing the central architecture
idea.

## Search provenance

The snapshot was collected from the public arXiv API on 2026-08-14 using
queries for `DFlash`, `speculative decoding`, `hardware`, `accelerator`, and
`GPU`. Paper claims and dates should be rechecked against the final DAC'27
submission deadline.

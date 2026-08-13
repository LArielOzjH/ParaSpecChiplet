# Literature boundary

Checked on 2026-08-14. Links are kept so later experiments can pin versions.

## Direct DFlash lineage

- [DFlash](https://arxiv.org/abs/2602.06036): block diffusion drafter, target hidden-feature injection, position-decayed loss, and a draft-depth/latency/acceptance trade-off.
- [DFlare](https://arxiv.org/abs/2606.02091): layer-wise fusion of a broader set of target layers to scale draft capacity. This makes layer-level heterogeneity especially important to distinguish from merely adding draft depth.
- [HyperDFlash](https://arxiv.org/abs/2606.26744): recent block-parallel draft-side representation changes; check against any proposed hidden-state or residual mechanism.

## Position, budget, and block-structure neighbors

- [D-PACE](https://arxiv.org/abs/2605.18810): dynamic position-aware cross-entropy for training, not a heterogeneous inference datapath.
- [D-Cut](https://arxiv.org/abs/2607.14647): cross-request adaptive verification-depth pruning with a runtime cost model. A ParaSpecChiplet schedule must retain all relevant draft positions or show a different mechanism from pruning.
- [BASTION](https://arxiv.org/abs/2605.29727): budget-aware tree construction and hardware-aware best-first expansion. Do not claim generic budget-aware speculation.
- [SpecBlock](https://arxiv.org/abs/2605.07243): block-iterative drafting, path dependence, rank-based branching, and cost-aware adaptation. This is a close algorithmic neighbor for any across-block or path-dependent interpretation.
- [SpecBound](https://arxiv.org/abs/2604.12247): layer-wise confidence calibration and bounded self-speculation. Distinguish DFlash's external block drafter and target-conditioned hidden features from self-draft early exit.
- [LibraSpec](https://arxiv.org/abs/2608.08721): training-free marginal-gain dynamic speculation length for diffusion-based drafters. Do not claim state-conditioned budget selection or expected-speedup optimization as algorithmic novelty.
- [CURE](https://arxiv.org/abs/2608.00531): uncertainty-focused repair for block-parallel drafting. Do not rebrand tail refinement or repair paths as a new algorithm; the differentiator must be physical execution/dataflow.
- [xPress](https://arxiv.org/abs/2608.02438): parallel causal refinement for diffusion drafters. Compare against it when discussing bidirectional-block interference and any refinement stage.
- [DBLAST](https://arxiv.org/abs/2608.05448): dependent block drafting and acceptance-oriented training for stochastic decoding. It strengthens the case that accepted-prefix behavior, not marginal position accuracy, is the relevant measurement.

## Deployment and architecture context

- [Windowed-MTP](https://arxiv.org/abs/2607.21535): shows that the cheap-drafter assumption fails at long context. Long-context evaluation must include this baseline and cannot claim sliding-window novelty.
- [SlimSpec](https://arxiv.org/abs/2605.10453): generic low-rank draft LM-head acceleration. This project does not claim LM-head compression.
- [EPIC](https://arxiv.org/abs/2606.00722): efficient CFG-constrained diffusion decoding. Grammar/sparse-head ideas are intentionally out of scope for the current thesis.
- [Compass](https://arxiv.org/abs/2512.06093): heterogeneous multi-chiplet mapping for LLM inference service workloads. It motivates explicit chiplet traffic and mapping baselines; it does not address DFlash survival-aware execution.
- [LaMoSys3.5D](https://arxiv.org/abs/2512.08731): 3D/chiplet LLM serving system context. Use it for memory/stacking assumptions, not as a DFlash baseline.
- [MemSpec](https://arxiv.org/abs/2608.10362): memory-aware runtime selection among adaptive draft models. It is a close systems neighbor for state-dependent scheduling, but it does not provide a DFlash layer-position execution fabric; use it to sharpen the hardware/dataflow boundary.

## Novelty test

A result is not a ParaSpecChiplet contribution if it can be described entirely as:

1. changing training position weights;
2. shortening or pruning the draft;
3. generic early exit without block-prefix survival accounting;
4. generic quantization or low-rank projection;
5. generic chiplet mapping without a DFlash-specific dataflow constraint.

The minimum architecture contribution must expose the layer-by-position survival map, a concrete schedule/dataflow, and a traffic-aware comparison against a monolithic design.

## Updated novelty warning

As of 2026-08-14, a paper whose main mechanism is only “observe recent
acceptance/confidence and choose a cheaper draft budget” is not sufficiently
distinct: LibraSpec, CURE, and MemSpec occupy nearby algorithm/runtime space.
ParaSpecChiplet must demonstrate a physical execution mechanism—shared lower
layers, protected-prefix/tail lane scheduling, activation multicast, and
mixed-depth queueing—that remains meaningful even when the policy is replaced
by a static oracle.

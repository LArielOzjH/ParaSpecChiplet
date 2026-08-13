#!/usr/bin/env python3
"""Measure prefix sensitivity to tail-state perturbation in a local DFlash model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.utils import select_anchors
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target = AutoModelForCausalLM.from_pretrained(
        str(args.target_dir), dtype=torch.float32, local_files_only=True
    )
    config = SpeculatorModelConfig.from_pretrained(
        str(args.draft_dir), local_files_only=True
    )
    draft = SpeculatorModel.from_pretrained(
        str(args.draft_dir), config=config, local_files_only=True, dtype=torch.float32
    )
    anchors_count = 8
    draft.config.max_anchors = anchors_count
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    text = (
        "The quick brown fox jumps over the lazy dog. In computer architecture, "
        "memory bandwidth and data movement often dominate modern neural network "
        "inference. A useful experiment should measure accuracy, latency, energy, "
        "and communication cost. "
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = torch.cat([input_ids, input_ids.flip(-1), input_ids], dim=1)[:, :126]
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    torch.manual_seed(7)
    anchors, _ = select_anchors(loss_mask, anchors_count, draft.block_size)

    with torch.inference_mode():
        target_output = target(
            input_ids=input_ids, output_hidden_states=True, use_cache=False
        )
        auxiliary = torch.cat(
            [target_output.hidden_states[index + 1] for index in draft.target_layer_ids],
            dim=-1,
        )

        def run(perturb_after: int | None) -> torch.Tensor:
            captured: list[torch.Tensor] = []
            handles = []
            for index, layer in enumerate(draft.layers):
                if index == perturb_after:
                    def perturb(_module, _args, output):
                        block = output.view(anchors_count, draft.block_size, -1).clone()
                        block[:, 4:] = block[:, 4:].mean(dim=1, keepdim=True)
                        return block.view_as(output)
                    handles.append(layer.register_forward_hook(perturb))
                if index == len(draft.layers) - 1:
                    handles.append(
                        layer.register_forward_hook(
                            lambda _module, _args, output: captured.append(
                                output.detach().clone()
                            )
                        )
                    )
            torch.manual_seed(7)
            DFlashDraftModel.forward.__wrapped__(
                draft,
                hidden_states=auxiliary,
                input_ids=input_ids,
                loss_mask=loss_mask,
                verifier_last_hidden_states=target_output.hidden_states[-1],
            )
            for handle in handles:
                handle.remove()
            return captured[0].view(anchors_count, draft.block_size, -1)

        baseline = run(None)[:, :4]
        results = []
        for perturb_after in (0, 1):
            perturbed = run(perturb_after)[:, :4]
            cosine = torch.nn.functional.cosine_similarity(baseline, perturbed, dim=-1)
            relative_l2 = (baseline - perturbed).norm(dim=-1) / (
                baseline.norm(dim=-1) + 1e-8
            )
            results.append(
                {
                    "perturb_after_layer": perturb_after + 1,
                    "prefix_cosine_by_position": cosine.mean(dim=0).tolist(),
                    "prefix_relative_l2_by_position": relative_l2.mean(dim=0).tolist(),
                    "mean_prefix_cosine": float(cosine.mean()),
                    "mean_prefix_relative_l2": float(relative_l2.mean()),
                }
            )

    output = {
        "anchors": anchors.tolist(),
        "tail_positions_replaced": "positions 5-8 with their per-block mean",
        "mapping_warning": "no token acceptance is inferred",
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()


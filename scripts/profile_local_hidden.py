#!/usr/bin/env python3
"""Profile DFlash block activations without relying on vocabulary mappings.

This intentionally measures representation geometry, not acceptance. The
checkpoint's draft-to-target vocabulary mapping must be available before token
agreement can be reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.utils import select_anchors
from transformers import AutoModelForCausalLM, AutoTokenizer


def profile(target_dir: Path, draft_dir: Path, text: str, anchors: int, max_length: int) -> dict:
    target = AutoModelForCausalLM.from_pretrained(
        str(target_dir), dtype=torch.float32, local_files_only=True
    )
    config = SpeculatorModelConfig.from_pretrained(str(draft_dir), local_files_only=True)
    draft = SpeculatorModel.from_pretrained(
        str(draft_dir), config=config, local_files_only=True, dtype=torch.float32
    )
    draft.config.max_anchors = anchors
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = torch.cat([input_ids, input_ids.flip(-1), input_ids], dim=1)[:, :max_length]
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    torch.manual_seed(7)
    selected_anchors, _ = select_anchors(loss_mask, anchors, draft.block_size)
    captured: list[torch.Tensor] = []
    for layer in draft.layers:
        layer.register_forward_hook(
            lambda _module, _args, output: captured.append(output.detach().clone())
        )

    with torch.inference_mode():
        target_output = target(
            input_ids=input_ids, output_hidden_states=True, use_cache=False
        )
        auxiliary = torch.cat(
            [target_output.hidden_states[index + 1] for index in draft.target_layer_ids],
            dim=-1,
        )
        torch.manual_seed(7)
        DFlashDraftModel.forward.__wrapped__(
            draft,
            hidden_states=auxiliary,
            input_ids=input_ids,
            loss_mask=loss_mask,
            verifier_last_hidden_states=target_output.hidden_states[-1],
        )

    layers = []
    for layer_index, output in enumerate(captured, start=1):
        block = output.view(anchors, draft.block_size, -1)
        normalized = block / (block.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = torch.einsum("bih,bjh->bij", normalized, normalized)
        off_diagonal = ~torch.eye(draft.block_size, dtype=torch.bool)
        adjacent = similarity[:, :-1, 1:].diagonal(dim1=1, dim2=2)
        layers.append(
            {
                "layer": layer_index,
                "mean_offdiag_cosine": float(similarity[:, off_diagonal].mean()),
                "mean_adjacent_cosine": float(adjacent.mean()),
                "position_mean_norm": [
                    round(float(value), 5) for value in block.norm(dim=-1).mean(dim=0)
                ],
            }
        )
    return {
        "model": draft_dir.name,
        "target": "Qwen/Qwen3-0.6B",
        "seq_len": int(input_ids.shape[1]),
        "anchors": selected_anchors.tolist(),
        "block_size": int(draft.block_size),
        "mapping_warning": "draft token IDs not compared; provide t2d/d2t before acceptance analysis",
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    result = profile(
        args.target_dir,
        args.draft_dir,
        "The quick brown fox jumps over the lazy dog. In computer architecture, memory bandwidth and data movement often dominate modern neural network inference. A useful experiment should measure accuracy, latency, energy, and communication cost. ",
        args.anchors,
        args.max_length,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


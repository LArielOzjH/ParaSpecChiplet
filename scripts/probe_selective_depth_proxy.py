#!/usr/bin/env python3
"""Measure prefix-agreement degradation from selective tail layer skipping.

This is a CPU/offline proxy. It still executes the skipped layer for all
positions and replaces selected tail outputs with their layer inputs, so it
does not claim compute speedup. Its purpose is to test whether a schedule is
safe enough to justify a real selective-depth kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.utils import select_anchors
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paraspec.offline_acceptance import accepted_prefix_length
from paraspec.selective_proxy import (
    selective_mlp_forward,
    skipped_positions,
    validate_depth_schedule,
)


DEFAULT_PROMPTS = (
    "Explain why memory bandwidth matters for neural network inference.",
    "Compare a monolithic accelerator with a chiplet-based accelerator.",
    "Give a concise definition of speculative decoding.",
    "Why can parallel token prediction be difficult to verify efficiently?",
)


def make_input_ids(tokenizer: AutoTokenizer, prompt: str, length: int = 126) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    pieces = [prompt_ids, prompt_ids.flip(-1)]
    while sum(piece.shape[1] for piece in pieces) < length:
        pieces.append(prompt_ids)
    return torch.cat(pieces, dim=1)[:, :length]


def load_prompts(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not prompts:
        raise ValueError("prompt file contains no non-empty lines")
    return prompts


def load_schedules(
    path: Path | None, block_size: int, draft_layers: int, protected_prefix: int
) -> dict[str, tuple[int, ...]]:
    if path is None:
        schedules = {
            "uniform": (draft_layers,) * block_size,
            "protected_staircase": (draft_layers, draft_layers)
            + tuple(max(1, draft_layers - index // 2) for index in range(block_size - 2)),
        }
    else:
        payload = json.loads(path.read_text())
        schedules = {str(name): tuple(int(value) for value in values) for name, values in payload.items()}
    return {
        name: validate_depth_schedule(
            values, draft_layers=draft_layers, protected_prefix=protected_prefix
        )
        for name, values in schedules.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--schedules", type=Path)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--protected-prefix", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("layer", "mlp"),
        default="layer",
        help="layer replaces the whole tail layer; mlp keeps bidirectional attention and skips only MLP updates",
    )
    args = parser.parse_args()

    target = AutoModelForCausalLM.from_pretrained(
        str(args.target_dir), dtype=torch.float32, local_files_only=True
    ).eval()
    config = SpeculatorModelConfig.from_pretrained(
        str(args.draft_dir), local_files_only=True
    )
    draft = SpeculatorModel.from_pretrained(
        str(args.draft_dir), config=config, local_files_only=True, dtype=torch.float32
    ).eval()
    draft.config.max_anchors = args.anchors
    schedules = load_schedules(
        args.schedules, draft.block_size, len(draft.layers), args.protected_prefix
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.target_dir), local_files_only=True)
    prompts = load_prompts(args.prompt_file)
    records: list[dict] = []

    for prompt_index, prompt in enumerate(prompts):
        input_ids = make_input_ids(tokenizer, prompt)
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        seed = args.seed + prompt_index
        torch.manual_seed(seed)
        anchors, anchor_valid = select_anchors(loss_mask, args.anchors, draft.block_size)
        with torch.inference_mode():
            target_output = target(
                input_ids=input_ids, output_hidden_states=True, use_cache=False
            )
            auxiliary = torch.cat(
                [target_output.hidden_states[index + 1] for index in draft.target_layer_ids],
                dim=-1,
            )
            target_tokens = target_output.logits.argmax(dim=-1)

            def run(schedule: tuple[int, ...]) -> torch.Tensor:
                handles = []
                for layer_index, layer in enumerate(draft.layers):
                    skipped = skipped_positions(schedule, layer_index=layer_index)

                    def layer_hook(module, layer_inputs, layer_kwargs, output, skipped=skipped):
                        layer_input = layer_kwargs.get("hidden_states")
                        if layer_input is None:
                            if not layer_inputs:
                                raise RuntimeError("DFlash layer hook received no hidden_states input")
                            layer_input = layer_inputs[0]
                        output_block = output.view(args.anchors, draft.block_size, -1).clone()
                        input_block = layer_input.view(args.anchors, draft.block_size, -1)
                        mask = torch.tensor(skipped, dtype=torch.bool, device=output.device)
                        output_block[:, mask] = input_block[:, mask]
                        return output_block.view_as(output)

                    def mlp_hook(module, layer_inputs, layer_kwargs, output, skipped=skipped):
                        hidden_states = layer_kwargs.get("hidden_states")
                        if hidden_states is None:
                            if not layer_inputs:
                                raise RuntimeError("DFlash MLP hook received no hidden_states input")
                            hidden_states = layer_inputs[0]
                        return selective_mlp_forward(
                            module,
                            hidden_states,
                            skipped=skipped,
                            anchors=args.anchors,
                            block_size=draft.block_size,
                        )

                    if any(skipped):
                        target_module = layer if args.mode == "layer" else layer.mlp
                        target_hook = layer_hook if args.mode == "layer" else mlp_hook
                        handles.append(target_module.register_forward_hook(target_hook, with_kwargs=True))
                torch.manual_seed(seed)
                draft_tokens, _, _ = DFlashDraftModel.forward.__wrapped__(
                    draft,
                    hidden_states=auxiliary,
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    verifier_last_hidden_states=target_output.hidden_states[-1],
                )
                for handle in handles:
                    handle.remove()
                return draft_tokens

            for schedule_name, schedule in schedules.items():
                draft_tokens = run(schedule)
                for anchor_index, anchor in enumerate(anchors.tolist()):
                    if not bool(anchor_valid[anchor_index]):
                        continue
                    draft_block = draft_tokens[
                        0, anchor_index * draft.block_size + 1 : (anchor_index + 1) * draft.block_size
                    ].tolist()
                    target_block = target_tokens[0, anchor : anchor + draft.block_size - 1].tolist()
                    records.append(
                        {
                            "kind": "dflash_selective_depth_offline_proxy",
                            "prompt_index": prompt_index,
                            "schedule": schedule_name,
                            "mode": args.mode,
                            "depth_by_position": schedule,
                            "anchor_position": int(anchor),
                            "accepted_prefix": accepted_prefix_length(draft_block, target_block),
                        }
                    )

    payload = {
        "kind": "dflash_selective_depth_offline_proxy_collection",
        "target": str(args.target_dir),
        "draft": str(args.draft_dir),
        "records": records,
        "warning": "skipped layers are still executed and replaced after the layer; this measures agreement degradation, not speedup",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompts": len(prompts), "schedules": len(schedules), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()

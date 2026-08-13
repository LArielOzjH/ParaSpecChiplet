#!/usr/bin/env python3
"""Probe DFlash prefix agreement on masked training-style blocks.

This is deliberately not the official autoregressive serving path. It uses the
Speculators training forward, fixes the random anchor seed so draft blocks are
aligned with target positions, and reports agreement against target greedy
predictions under the original context.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
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
    tokenizer = AutoTokenizer.from_pretrained(str(args.target_dir), local_files_only=True)
    prompts = load_prompts(args.prompt_file)
    events: list[dict] = []

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
            # The draft forward samples anchors internally; reset the seed to
            # reproduce exactly the anchor order used above.
            torch.manual_seed(seed)
            draft_tokens, _, _ = DFlashDraftModel.forward.__wrapped__(
                draft,
                hidden_states=auxiliary,
                input_ids=input_ids,
                loss_mask=loss_mask,
                verifier_last_hidden_states=target_output.hidden_states[-1],
            )
            target_tokens = target_output.logits.argmax(dim=-1)

        for anchor_index, anchor in enumerate(anchors.tolist()):
            if not bool(anchor_valid[anchor_index]):
                continue
            draft_start = anchor_index * draft.block_size + 1
            draft_end = (anchor_index + 1) * draft.block_size
            target_start = anchor
            target_end = anchor + draft.block_size - 1
            draft_block = draft_tokens[0, draft_start:draft_end].tolist()
            target_block = target_tokens[0, target_start:target_end].tolist()
            events.append(
                {
                    "kind": "dflash_offline_acceptance_proxy",
                    "prompt_index": prompt_index,
                    "anchor_index": anchor_index,
                    "anchor_position": int(anchor),
                    "block_size": draft.block_size,
                    "accepted_prefix": accepted_prefix_length(draft_block, target_block),
                    "draft_tokens": draft_block,
                    "target_tokens": target_block,
                    "seed": seed,
                }
            )

    payload = {
        "kind": "dflash_offline_acceptance_proxy_collection",
        "target": str(args.target_dir),
        "draft": str(args.draft_dir),
        "events": events,
        "warning": "masked training-style agreement under fixed context; not official autoregressive serving acceptance",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompts": len(prompts), "events": len(events)}, indent=2))


if __name__ == "__main__":
    main()

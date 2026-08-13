#!/usr/bin/env python3
"""Run a multi-condition tail-to-prefix perturbation sweep on local DFlash."""

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

# Make direct ``python scripts/...py`` invocation behave like a module run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paraspec.tail_perturbation import (
    PERTURBATION_MODES,
    perturb_tail,
    summarize_perturbation_records,
)


DEFAULT_PROMPTS = (
    "Explain why memory bandwidth matters for neural network inference.",
    "Compare a monolithic accelerator with a chiplet-based accelerator.",
    "Give a concise definition of speculative decoding.",
    "Why can parallel token prediction be difficult to verify efficiently?",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--protected-prefixes", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--modes", nargs="+", choices=PERTURBATION_MODES, default=list(PERTURBATION_MODES))
    return parser.parse_args()


def load_prompts(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not prompts:
        raise ValueError("prompt file contains no non-empty lines")
    return prompts


def make_input_ids(tokenizer: AutoTokenizer, prompt: str, length: int = 126) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    pieces = [prompt_ids, prompt_ids.flip(-1)]
    while sum(piece.shape[1] for piece in pieces) < length:
        pieces.append(prompt_ids)
    return torch.cat(pieces, dim=1)[:, :length]


def run_sweep(args: argparse.Namespace) -> dict:
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target_dir), dtype=torch.float32, local_files_only=True
    )
    config = SpeculatorModelConfig.from_pretrained(
        str(args.draft_dir), local_files_only=True
    )
    draft = SpeculatorModel.from_pretrained(
        str(args.draft_dir), config=config, local_files_only=True, dtype=torch.float32
    )
    draft.config.max_anchors = args.anchors
    tokenizer = AutoTokenizer.from_pretrained(str(args.target_dir), local_files_only=True)
    prompts = load_prompts(args.prompt_file)

    all_records: list[dict] = []
    per_prompt: list[dict] = []
    for prompt_index, prompt in enumerate(prompts):
        input_ids = make_input_ids(tokenizer, prompt)
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        torch.manual_seed(7 + prompt_index)
        anchors, _ = select_anchors(loss_mask, args.anchors, draft.block_size)

        with torch.inference_mode():
            target_output = target(
                input_ids=input_ids, output_hidden_states=True, use_cache=False
            )
            auxiliary = torch.cat(
                [target_output.hidden_states[index + 1] for index in draft.target_layer_ids],
                dim=-1,
            )

            def run(perturb_after: int | None, mode: str | None, protected_prefix: int | None) -> torch.Tensor:
                captured: list[torch.Tensor] = []
                handles = []
                for index, layer in enumerate(draft.layers):
                    if perturb_after == index + 1:
                        def perturb(_module, _args, output):
                            block = output.view(args.anchors, draft.block_size, -1)
                            return perturb_tail(
                                block,
                                protected_prefix=protected_prefix or draft.block_size,
                                mode=mode or "mean",
                            ).view_as(output)

                        handles.append(layer.register_forward_hook(perturb))
                    if index == len(draft.layers) - 1:
                        handles.append(
                            layer.register_forward_hook(
                                lambda _module, _args, output: captured.append(output.detach().clone())
                            )
                        )
                torch.manual_seed(7 + prompt_index)
                DFlashDraftModel.forward.__wrapped__(
                    draft,
                    hidden_states=auxiliary,
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    verifier_last_hidden_states=target_output.hidden_states[-1],
                )
                for handle in handles:
                    handle.remove()
                return captured[0].view(args.anchors, draft.block_size, -1)

            baseline = run(None, None, None)[:, :]
            prompt_records = []
            for after_layer in range(1, len(draft.layers)):
                for protected_prefix in args.protected_prefixes:
                    if not 0 <= protected_prefix <= draft.block_size:
                        raise ValueError("protected prefix must be within the draft block")
                    for mode in args.modes:
                        perturbed = run(after_layer, mode, protected_prefix)
                        cosine = torch.nn.functional.cosine_similarity(
                            baseline[:, :protected_prefix],
                            perturbed[:, :protected_prefix],
                            dim=-1,
                        )
                        relative_l2 = (
                            baseline[:, :protected_prefix] - perturbed[:, :protected_prefix]
                        ).norm(dim=-1) / (
                            baseline[:, :protected_prefix].norm(dim=-1) + 1e-8
                        )
                        record = {
                            "prompt_index": prompt_index,
                            "after_layer": after_layer,
                            "protected_prefix": protected_prefix,
                            "mode": mode,
                            "cosine": float(cosine.mean()),
                            "relative_l2": float(relative_l2.mean()),
                            "cosine_by_position": cosine.mean(dim=0).tolist(),
                            "relative_l2_by_position": relative_l2.mean(dim=0).tolist(),
                        }
                        all_records.append(record)
                        prompt_records.append(record)
        per_prompt.append({"prompt_index": prompt_index, "prompt": prompt, "records": prompt_records})

    return {
        "model": {"target": str(args.target_dir), "draft": str(args.draft_dir)},
        "block_size": draft.block_size,
        "anchors": args.anchors,
        "prompts": list(prompts),
        "results": all_records,
        "summary": summarize_perturbation_records(all_records),
        "mapping_warning": "no token acceptance is inferred; this is activation geometry only",
        "tail_definition": "positions after protected_prefix are perturbed after the selected draft layer",
        "per_prompt": per_prompt,
    }


def main() -> None:
    args = parse_args()
    output = run_sweep(args)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()

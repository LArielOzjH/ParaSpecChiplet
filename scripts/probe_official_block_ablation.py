#!/usr/bin/env python3
"""Measure official DFlash acceptance after bypassing draft Transformer blocks.

The target verifier is unchanged. This probe reports acceptance and measured
serving-loop latency only; it does not claim that a Python hook is a hardware
implementation or a speedup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paraspec.block_ablation import (
    install_layer_bypasses,
    install_mlp_bypasses,
    parse_layer_groups,
    validate_layer_indices,
)
from paraspec.official_trace import stats_to_verification_events


def load_prompts(path: Path) -> tuple[str, ...]:
    prompts = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not prompts:
        raise ValueError("prompt file contains no non-empty lines")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--layers", default=None, help="comma-separated zero-based layers; default: all")
    parser.add_argument(
        "--groups",
        default=None,
        help="semicolon-separated layer groups, e.g. 2,3;2,4;3,4",
    )
    parser.add_argument(
        "--mode",
        choices=("layer", "mlp"),
        default="layer",
        help="bypass a whole draft block or only its MLP update",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("official DFlash block ablation requires CUDA")

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path.cwd()))
    from dflash.model import dflash_generate

    dtype = torch.bfloat16
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target_model), dtype=dtype
    ).eval().to(args.device)
    draft = AutoModel.from_pretrained(
        str(args.draft_model), trust_remote_code=True, dtype=dtype
    ).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.target_model))
    prompts = load_prompts(args.prompts)
    draft_layers = len(draft.layers)
    if args.groups is not None and args.layers is not None:
        raise ValueError("use only one of --layers and --groups")
    if args.groups is not None:
        groups = parse_layer_groups(args.groups, draft_layers=draft_layers)
    elif args.layers is None:
        groups = tuple(
            (f"bypass_layer_{index}", (index,)) for index in range(draft_layers)
        )
    else:
        layer_indices = validate_layer_indices(
            [int(value) for value in args.layers.split(",") if value.strip()],
            draft_layers=draft_layers,
        )
        groups = tuple(
            (f"bypass_layer_{index}", (index,)) for index in layer_indices
        )

    experiments: tuple[tuple[str, tuple[int, ...]], ...] = (("uniform", ()),) + groups
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    records: list[dict] = []

    for prompt_index, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        for experiment_name, ablated_layers in experiments:
            restore = None
            if ablated_layers:
                installer = install_layer_bypasses if args.mode == "layer" else install_mlp_bypasses
                restore = installer(draft.layers, ablated_layers, draft_layers=draft_layers)
            try:
                stats = dflash_generate(
                    draft,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_token_ids,
                    temperature=0.0,
                    block_size=int(draft.block_size),
                    return_stats=True,
                )
            finally:
                if restore is not None:
                    restore()

            events = stats_to_verification_events(
                request_id=f"prompt-{prompt_index}",
                block_size=int(draft.block_size),
                committed_tokens_per_cycle=stats.acceptance_lengths,
                draft_layers=draft_layers,
                stage_latency_us={"end_to_end_avg": float(stats.time_per_output_token * 1e6)},
            )
            for event in events:
                event.update(
                    {
                        "kind": "dflash_block_ablation",
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "experiment": experiment_name,
                        "ablation_mode": args.mode,
                        "ablated_layers": list(ablated_layers),
                        "target_model": str(args.target_model),
                        "draft_model": str(args.draft_model),
                    }
                )
                records.append(event)

    args.output.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(json.dumps({"events": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture lossless DFlash acceptance lengths using the official implementation.

This is intentionally a GPU-only adapter: the upstream timing helper calls
CUDA synchronization. It records accepted draft prefixes, not activation
checkpoints or token mappings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paraspec.official_trace import stats_to_verification_events
from paraspec.capture_config import resolve_block_size


DEFAULT_PROMPTS = (
    "Explain why memory bandwidth matters for neural network inference.",
    "Compare a monolithic accelerator with a chiplet-based accelerator.",
    "Give a concise definition of speculative decoding.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dflash-repo", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Optional override; must match the draft checkpoint block size.",
    )
    return parser.parse_args()


def load_prompts(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not prompts:
        raise ValueError("prompt file contains no non-empty lines")
    return prompts


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("official DFlash trace capture requires CUDA")
    sys.path.insert(0, str(args.dflash_repo))
    from dflash.model import dflash_generate

    dtype = torch.bfloat16
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        device_map=args.device,
    ).eval()
    draft = AutoModel.from_pretrained(
        args.draft_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    prompts = load_prompts(args.prompt_file)
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    block_size = resolve_block_size(args.block_size, int(draft.block_size))

    events: list[dict] = []
    for prompt_index, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        stats = dflash_generate(
            draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=0.0,
            block_size=block_size,
            return_stats=True,
        )
        avg_us = float(stats.time_per_output_token * 1e6)
        prompt_events = stats_to_verification_events(
            request_id=f"prompt-{prompt_index}",
            block_size=args.block_size,
            committed_tokens_per_cycle=stats.acceptance_lengths,
            draft_layers=len(draft.layers),
            stage_latency_us={"end_to_end_avg": avg_us},
        )
        for event in prompt_events:
            event.update(
                {
                    "prompt": prompt,
                    "target_model": args.target_model,
                    "draft_model": args.draft_model,
                    "num_output_tokens": int(stats.num_output_tokens),
                    "time_to_first_token_us": float(stats.time_to_first_token * 1e6),
                    "time_per_output_token_us": avg_us,
                }
            )
            events.append(event)

    args.output.write_text("".join(json.dumps(event) + "\n" for event in events))
    print(json.dumps({"events": len(events), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure official DFlash acceptance under attention-preserving MLP gating.

This is an acceptance experiment, not a claim of end-to-end speedup. The
official DFlash verifier remains unchanged; only the draft MLP calls are made
row-selective according to a fixed per-position depth schedule.
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

from paraspec.official_selective import install_mlp_gates
from paraspec.official_trace import stats_to_verification_events


def load_prompts(path: Path) -> tuple[str, ...]:
    prompts = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not prompts:
        raise ValueError("prompt file contains no non-empty lines")
    return prompts


def load_schedules(path: Path, block_size: int, draft_layers: int) -> dict[str, tuple[int, ...]]:
    payload = json.loads(path.read_text())
    schedules = {str(name): tuple(int(value) for value in values) for name, values in payload.items()}
    for name, schedule in schedules.items():
        if len(schedule) != block_size:
            raise ValueError(f"schedule {name!r} must contain {block_size} positions")
        if any(depth < 1 or depth > draft_layers for depth in schedule):
            raise ValueError(f"schedule {name!r} has an invalid draft depth")
    return schedules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--schedules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("official DFlash gating probe requires CUDA")
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
    schedules = load_schedules(args.schedules, int(draft.block_size), len(draft.layers))
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    records: list[dict] = []

    for prompt_index, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        for schedule_name, schedule in schedules.items():
            restore = None
            if any(depth != len(draft.layers) for depth in schedule):
                restore = install_mlp_gates(
                    draft.layers, schedule, draft_layers=len(draft.layers)
                )
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
                draft_layers=len(draft.layers),
                stage_latency_us={"end_to_end_avg": float(stats.time_per_output_token * 1e6)},
            )
            for event in events:
                event.update(
                    {
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "schedule": schedule_name,
                        "depth_by_position": list(schedule),
                        "mlp_work": sum(schedule),
                        "target_model": str(args.target_model),
                        "draft_model": str(args.draft_model),
                    }
                )
                records.append(event)

    args.output.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(json.dumps({"events": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

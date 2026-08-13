#!/usr/bin/env python3
"""Profile official DFlash draft attention and MLP stage time with CUDA events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dflash-repo", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import sys

    sys.path.insert(0, str(args.dflash_repo))
    from dflash.model import dflash_generate

    dtype = torch.bfloat16
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target_model), dtype=dtype, device_map=args.device
    ).eval()
    draft = AutoModel.from_pretrained(
        str(args.draft_model), trust_remote_code=True, dtype=dtype, device_map=args.device
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.target_model))
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    starts: dict[str, list[torch.cuda.Event]] = {}
    ends: dict[str, list[torch.cuda.Event]] = {}
    handles = []

    def attach(module: object, name: str) -> None:
        starts[name] = []
        ends[name] = []

        def before(*_: object) -> None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            starts[name].append(event)

        def after(*_: object) -> None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            ends[name].append(event)

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))

    for layer_index, layer in enumerate(draft.layers):
        attach(layer.self_attn, f"layer{layer_index}.attention")
        attach(layer.mlp, f"layer{layer_index}.mlp")

    try:
        with torch.inference_mode():
            stats = dflash_generate(
                draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
                temperature=0.0,
                block_size=int(draft.block_size),
                return_stats=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    torch.cuda.synchronize()

    stage_ms = {}
    for name in starts:
        if len(starts[name]) != len(ends[name]):
            raise RuntimeError(f"unbalanced CUDA events for {name}")
        stage_ms[name] = {
            "calls": len(starts[name]),
            "total_ms": sum(start.elapsed_time(end) for start, end in zip(starts[name], ends[name])),
        }

    payload = {
        "target_model": str(args.target_model),
        "draft_model": str(args.draft_model),
        "prompt": args.prompt,
        "block_size": int(draft.block_size),
        "draft_layers": len(draft.layers),
        "accepted_cycles": len(stats.acceptance_lengths),
        "time_per_output_token_us": float(stats.time_per_output_token * 1e6),
        "stage_ms": stage_ms,
        "warning": "CUDA hook instrumentation adds event overhead; stage totals are directional, not end-to-end latency",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"stages": len(stage_ms), "cycles": payload["accepted_cycles"]}, indent=2))


if __name__ == "__main__":
    main()

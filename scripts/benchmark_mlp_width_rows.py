#!/usr/bin/env python3
"""Benchmark reduced-width gated MLP execution on a real draft model.

This is a CUDA microbenchmark, not an end-to-end serving result. Reduced
intermediate width is a hardware proxy for partial MLP fidelity/reduced lanes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paraspec.partial_mlp import reduced_gated_mlp


def benchmark(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) * 1000 / iterations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--active-per-request", type=int, nargs="+", default=(16, 12, 9, 8, 4))
    parser.add_argument("--fractions", type=float, nargs="+", default=(1.0, 0.75, 0.5, 0.25))
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(args.draft_model), trust_remote_code=True, dtype=torch.bfloat16
    ).eval().to(args.device)
    mlp = model.layers[0].mlp
    block_size = int(model.block_size)
    hidden_size = int(model.config.hidden_size)
    intermediate_size = int(mlp.gate_proj.weight.shape[0])
    values = torch.randn(
        args.batch_size, block_size, hidden_size, device=args.device, dtype=torch.bfloat16
    )
    flat = values.reshape(-1, hidden_size)
    records = []

    for active_per_request in args.active_per_request:
        if not 1 <= active_per_request <= block_size:
            raise ValueError("active rows per request must be within block size")
        mask = torch.zeros(
            args.batch_size, block_size, device=args.device, dtype=torch.bool
        )
        mask[:, :active_per_request] = True
        active = flat[mask.reshape(-1)]
        for fraction in args.fractions:
            if not 0.0 < fraction <= 1.0:
                raise ValueError("fractions must be within (0, 1]")
            width = max(1, int(intermediate_size * fraction))

            def dense():
                return mlp(active)

            def reduced():
                return reduced_gated_mlp(active, mlp, width)

            records.append(
                {
                    "active_per_request": active_per_request,
                    "active_rows": int(active.shape[0]),
                    "width_fraction": fraction,
                    "intermediate_width": width,
                    "dense_ms": benchmark(dense, warmup=args.warmup, iterations=args.iterations),
                    "reduced_ms": benchmark(reduced, warmup=args.warmup, iterations=args.iterations),
                }
            )

    payload = {
        "draft_model": str(args.draft_model),
        "batch_size": args.batch_size,
        "block_size": block_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "warning": "reduced-width CUDA microbenchmark; not acceptance or end-to-end serving speedup",
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"records": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

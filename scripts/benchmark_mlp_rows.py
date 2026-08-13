#!/usr/bin/env python3
"""Benchmark dense versus active-row MLP execution for a DFlash draft model."""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModel


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
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--active-per-request", type=int, nargs="+", default=(16, 12, 9, 8, 4, 2))
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=75)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model = AutoModel.from_pretrained(
        args.draft_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()
    mlp = model.layers[0].mlp
    block_size = int(model.block_size)
    hidden_size = int(model.config.hidden_size)
    values = torch.randn(
        args.batch_size, block_size, hidden_size, device=args.device, dtype=torch.bfloat16
    )
    flat = values.reshape(-1, hidden_size)
    output = torch.zeros_like(flat)
    records = []

    for active_per_request in args.active_per_request:
        if not 1 <= active_per_request <= block_size:
            raise ValueError("active rows per request must be within block size")
        mask = torch.zeros(
            args.batch_size, block_size, device=args.device, dtype=torch.bool
        )
        mask[:, :active_per_request] = True
        mask = mask.reshape(-1)
        active = flat[mask]

        def dense():
            return mlp(values)

        def active_only():
            return mlp(active)

        def grouped():
            result = mlp(active)
            output.zero_()
            output[mask] = result
            return output

        records.append(
            {
                "active_per_request": active_per_request,
                "active_rows": int(active.shape[0]),
                "dense_ms": benchmark(dense, warmup=args.warmup, iterations=args.iterations),
                "active_ms": benchmark(active_only, warmup=args.warmup, iterations=args.iterations),
                "grouped_ms": benchmark(grouped, warmup=args.warmup, iterations=args.iterations),
            }
        )

    payload = {
        "draft_model": args.draft_model,
        "batch_size": args.batch_size,
        "block_size": block_size,
        "hidden_size": hidden_size,
        "warning": "single-layer CUDA microbenchmark; not end-to-end serving speedup",
        "records": records,
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"records": len(records), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()

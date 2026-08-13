#!/usr/bin/env python3
"""Benchmark graph-safe persistent replay for a fixed grouped MLP shape."""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModel


def benchmark(fn, warmup: int, iterations: int) -> float:
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
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 8, 64))
    parser.add_argument("--active-per-request", type=int, default=9)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=60)
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
    records = []
    for batch_size in args.batch_sizes:
        if not 1 <= args.active_per_request <= block_size:
            raise ValueError("active-per-request must be within block size")
        values = torch.randn(
            batch_size, block_size, hidden_size, device=args.device, dtype=torch.bfloat16
        )
        flat = values.reshape(-1, hidden_size)
        mask = torch.zeros(batch_size, block_size, device=args.device, dtype=torch.bool)
        mask[:, : args.active_per_request] = True
        mask = mask.reshape(-1)
        indices = torch.where(mask)[0]
        active = flat[mask]
        output = torch.zeros_like(flat)
        zeros = torch.zeros_like(flat)

        def eager_grouped():
            result = mlp(active)
            output.copy_(zeros)
            output.index_copy_(0, indices, result)
            return output

        for _ in range(10):
            eager_grouped()
        torch.cuda.synchronize()
        static_active = active.clone()
        static_output = torch.zeros_like(output)
        static_zeros = torch.zeros_like(output)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            result = mlp(static_active)
            static_output.copy_(static_zeros)
            static_output.index_copy_(0, indices, result)

        def replay():
            graph.replay()
            return static_output

        records.append(
            {
                "batch_size": batch_size,
                "active_per_request": args.active_per_request,
                "eager_grouped_ms": benchmark(eager_grouped, args.warmup, args.iterations),
                "cuda_graph_ms": benchmark(replay, args.warmup, args.iterations),
                "dense_ms": benchmark(lambda: mlp(values), args.warmup, args.iterations),
            }
        )
        del graph
        torch.cuda.empty_cache()

    payload = {
        "draft_model": args.draft_model,
        "block_size": block_size,
        "hidden_size": hidden_size,
        "warning": "single-layer CUDA graph microbenchmark; not end-to-end serving speedup",
        "records": records,
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"records": len(records), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()

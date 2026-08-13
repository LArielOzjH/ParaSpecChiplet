#!/usr/bin/env python3
"""Proof-of-concept Triton gated MLP for fixed reduced widths.

This benchmark is intentionally separate from the official acceptance probe.
It checks numerical agreement and kernel latency only; it is not an
end-to-end serving result.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def gate_up_kernel(
        x_ptr,
        gate_ptr,
        up_ptr,
        gate_out_ptr,
        up_out_ptr,
        m,
        k,
        n,
        stride_xm,
        stride_gn,
        stride_un,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, k, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0,
            ).to(tl.float32)
            gate = tl.load(
                gate_ptr + offs_n[:, None] * stride_gn + offs_k[None, :],
                mask=(offs_n[:, None] < n) & (offs_k[None, :] < k),
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                up_ptr + offs_n[:, None] * stride_un + offs_k[None, :],
                mask=(offs_n[:, None] < n) & (offs_k[None, :] < k),
                other=0.0,
            ).to(tl.float32)
            acc_gate += tl.dot(x, tl.trans(gate))
            acc_up += tl.dot(x, tl.trans(up))
        out_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
        tl.store(
            gate_out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc_gate,
            mask=out_mask,
        )
        tl.store(
            up_out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc_up,
            mask=out_mask,
        )

    @triton.jit
    def down_kernel(
        gate_ptr,
        up_ptr,
        down_ptr,
        out_ptr,
        m,
        n,
        h,
        stride_gm,
        stride_gn,
        stride_dn,
        stride_dh,
        stride_om,
        stride_oh,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
        for n_start in range(0, n, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            gate = tl.load(
                gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0,
            )
            up = tl.load(
                up_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0,
            )
            down = tl.load(
                down_ptr + offs_h[:, None] * stride_dh + offs_n[None, :] * stride_dn,
                mask=(offs_h[:, None] < h) & (offs_n[None, :] < n),
                other=0.0,
            )
            down = down.to(tl.float32)
            activated = gate * tl.sigmoid(gate) * up
            acc += tl.dot(activated, tl.trans(down))
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh,
            acc,
            mask=(offs_m[:, None] < m) & (offs_h[None, :] < h),
        )

    return gate_up_kernel, down_kernel


def triton_gated_mlp(hidden_states, mlp, width: int):
    import triton

    gate_up_kernel, down_kernel = _kernels()
    m, k = hidden_states.shape
    gate_weight = mlp.gate_proj.weight[:width].contiguous()
    up_weight = mlp.up_proj.weight[:width].contiguous()
    down_weight = mlp.down_proj.weight[:, :width].contiguous()
    gate = torch.empty((m, width), device=hidden_states.device, dtype=torch.float32)
    up = torch.empty_like(gate)
    output = torch.empty((m, k), device=hidden_states.device, dtype=torch.float32)
    gate_up_kernel[(triton.cdiv(m, 32), triton.cdiv(width, 64))](
        hidden_states,
        gate_weight,
        up_weight,
        gate,
        up,
        m,
        k,
        width,
        k,
        k,
        k,
        width,
        1,
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=2,
        num_stages=2,
    )
    down_kernel[(triton.cdiv(m, 32), triton.cdiv(k, 64))](
        gate,
        up,
        down_weight,
        output,
        m,
        width,
        k,
        width,
        1,
        1,
        width,
        k,
        1,
        BLOCK_M=32,
        BLOCK_H=64,
        BLOCK_N=64,
        num_warps=2,
        num_stages=2,
    )
    return output.to(hidden_states.dtype)


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--active-per-request", type=int, nargs="+", default=(16, 9, 4))
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(args.draft_model), trust_remote_code=True, dtype=torch.bfloat16
    ).eval().cuda()
    mlp = model.layers[0].mlp
    block_size = int(model.block_size)
    hidden_size = int(model.config.hidden_size)
    intermediate_size = int(mlp.gate_proj.weight.shape[0])
    width = int(intermediate_size * args.fraction)
    records = []
    for active_per_request in args.active_per_request:
        values = torch.randn(
            args.batch_size * active_per_request,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        reference = lambda: F.linear(
            F.silu(F.linear(values, mlp.gate_proj.weight[:width]))
            * F.linear(values, mlp.up_proj.weight[:width]),
            mlp.down_proj.weight[:, :width],
        )
        triton_fn = lambda: triton_gated_mlp(values, mlp, width)
        triton_out = triton_fn()
        max_error = float(
            (triton_out - reference()).abs().max().float().cpu()
        )
        records.append(
            {
                "active_per_request": active_per_request,
                "active_rows": int(values.shape[0]),
                "width_fraction": args.fraction,
                "max_abs_error": max_error,
                "reference_ms": benchmark(reference, warmup=args.warmup, iterations=args.iterations),
                "triton_ms": benchmark(triton_fn, warmup=args.warmup, iterations=args.iterations),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "draft_model": str(args.draft_model),
                "batch_size": args.batch_size,
                "block_size": block_size,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "warning": "Triton microbenchmark; not acceptance or end-to-end serving speedup",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"records": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

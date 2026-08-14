#!/usr/bin/env python3
"""Sweep the parallel gain required for chiplet specialization to win."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paraspec.chiplet_boundary import chiplet_break_even_parallel_gain
from paraspec.chiplet_cost import estimate_chiplet_width_aware_mlp_cost, estimate_width_aware_mlp_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--draft-layers", type=int, default=5)
    parser.add_argument("--attention-macs", type=int, default=26_214_400)
    parser.add_argument("--mlp-macs", type=int, default=74_711_040)
    parser.add_argument("--compute-macs-per-cycle", type=int, default=1_000_000)
    args = parser.parse_args()

    rows = []
    for width in (1.0, 0.75, 0.5, 0.25):
        widths = (1.0,) * (args.draft_layers - 1) + (width,)
        mono = estimate_width_aware_mlp_cost(
            widths,
            block_size=args.block_size,
            attention_macs_per_layer=args.attention_macs,
            mlp_macs_per_layer=args.mlp_macs,
            compute_macs_per_cycle=args.compute_macs_per_cycle,
        )
        for bandwidth in (128, 256, 512, 1024):
            chiplet = estimate_chiplet_width_aware_mlp_cost(
                widths,
                block_size=args.block_size,
                attention_macs_per_layer=args.attention_macs,
                mlp_macs_per_layer=args.mlp_macs,
                compute_macs_per_cycle=args.compute_macs_per_cycle,
                activation_bytes_per_position=4096,
                link_bytes_per_cycle=bandwidth,
                synchronization_cycles=20,
                activation_multicast_reuse=4,
                router_cycles_per_position=0.25,
            )
            overhead = chiplet.total_cycles - chiplet.compute_cycles
            rows.append(
                {
                    "reduced_width": width,
                    "link_bytes_per_cycle": bandwidth,
                    "monolithic_compute_cycles": mono.compute_cycles,
                    "chiplet_compute_cycles": chiplet.compute_cycles,
                    "chiplet_overhead_cycles": overhead,
                    "break_even_parallel_gain": chiplet_break_even_parallel_gain(
                        compute_cycles=chiplet.compute_cycles,
                        overhead_cycles=overhead,
                    ),
                }
            )

    args.output.write_text(
        json.dumps(
            {
                "kind": "chiplet_break_even_parallel_gain",
                "warning": "analytical equal-resource boundary; not chiplet hardware timing",
                "parameters": vars(args) | {"output": str(args.output)},
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()

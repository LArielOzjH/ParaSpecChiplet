#!/usr/bin/env python3
"""Sweep width-aware monolithic/chiplet analytical costs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paraspec.chiplet_cost import (
    estimate_chiplet_width_aware_mlp_cost,
    estimate_width_aware_mlp_cost,
)


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
            synchronization_cycles=0,
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
            rows.append(
                {
                    "reduced_width": width,
                    "link_bytes_per_cycle": bandwidth,
                    "monolithic_cycles": mono.total_cycles,
                    "chiplet_cycles": chiplet.total_cycles,
                    "chiplet_over_monolithic": chiplet.total_cycles / mono.total_cycles,
                }
            )
    args.output.write_text(
        json.dumps(
            {
                "kind": "width_aware_fabric_sweep",
                "warning": "analytical calibrated sweep; not measured chiplet or end-to-end speedup",
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

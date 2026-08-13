#!/usr/bin/env python3
"""Sweep chiplet link/reuse assumptions against a monolithic baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paraspec.fabric_sweep import sweep_chiplet_tradeoff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-bandwidths", type=float, nargs="+", required=True)
    parser.add_argument("--reuse", type=float, nargs="+", required=True)
    parser.add_argument("--macs-per-layer", type=int, required=True)
    parser.add_argument("--compute-macs-per-cycle", type=int, required=True)
    parser.add_argument("--activation-bytes-per-position", type=int, required=True)
    parser.add_argument("--synchronization-cycles", type=int, default=0)
    parser.add_argument("--shared-lower-depth", type=int, default=0)
    parser.add_argument("--router-cycles-per-position", type=float, default=0.0)
    args = parser.parse_args()

    points = sweep_chiplet_tradeoff(
        depth_by_position=tuple(args.depths),
        base_cost_kwargs={
            "macs_per_layer": args.macs_per_layer,
            "compute_macs_per_cycle": args.compute_macs_per_cycle,
            "activation_bytes_per_position": args.activation_bytes_per_position,
            "synchronization_cycles": args.synchronization_cycles,
            "shared_lower_depth": args.shared_lower_depth,
            "router_cycles_per_position": args.router_cycles_per_position,
        },
        link_bytes_per_cycle_values=args.link_bandwidths,
        multicast_reuse_values=args.reuse,
    )
    payload = {
        "depth_by_position": args.depths,
        "warning": "transparent analytical model; not measured hardware speedup",
        "points": [point.__dict__ for point in points],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"points": len(points)}, indent=2))


if __name__ == "__main__":
    main()

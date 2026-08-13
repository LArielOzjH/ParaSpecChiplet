#!/usr/bin/env python3
"""Enumerate a survival/cost schedule frontier from an ablation trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paraspec.ablation import LayerPositionTrace
from paraspec.frontier import enumerate_depth_schedules, pareto_frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--protected-prefix", type=int, default=4)
    parser.add_argument("--macs-per-layer", type=int, required=True)
    parser.add_argument("--compute-macs-per-cycle", type=int, required=True)
    parser.add_argument("--activation-bytes-per-position", type=int, required=True)
    parser.add_argument("--link-bytes-per-cycle", type=int, required=True)
    parser.add_argument("--synchronization-cycles", type=int, required=True)
    parser.add_argument("--shared-lower-depth", type=int, default=0)
    parser.add_argument("--activation-multicast-reuse", type=float, default=1.0)
    parser.add_argument("--router-cycles-per-position", type=float, default=0.0)
    args = parser.parse_args()

    trace = LayerPositionTrace.from_mapping(json.loads(args.trace.read_text()))
    schedules = enumerate_depth_schedules(
        block_size=trace.block_size,
        draft_layers=trace.draft_layers,
        min_depth=args.min_depth,
        protected_prefix=args.protected_prefix,
    )
    points = pareto_frontier(
        trace,
        depth_schedules=schedules,
        cost_kwargs={
            "macs_per_layer": args.macs_per_layer,
            "compute_macs_per_cycle": args.compute_macs_per_cycle,
            "activation_bytes_per_position": args.activation_bytes_per_position,
            "link_bytes_per_cycle": args.link_bytes_per_cycle,
            "synchronization_cycles": args.synchronization_cycles,
            "shared_lower_depth": args.shared_lower_depth,
            "activation_multicast_reuse": args.activation_multicast_reuse,
            "router_cycles_per_position": args.router_cycles_per_position,
        },
    )
    payload = {
        "prediction_warning": "survival is additive layer-gain proxy; validate each candidate with real selective-depth traces",
        "candidate_count": len(schedules),
        "frontier": [
            {
                "depth_by_position": point.depth_by_position,
                "predicted_survival": point.predicted_survival,
                "predicted_value": point.predicted_value,
                "cost": point.cost.__dict__,
            }
            for point in points
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"candidates": len(schedules), "frontier": len(points)}, indent=2))


if __name__ == "__main__":
    main()

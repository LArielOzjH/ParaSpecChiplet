#!/usr/bin/env python3
"""Sweep grouped-vs-dense cost across heterogeneous schedule occupancy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paraspec.queue_model import analyze_schedule_queue


UNIFORM = (5,) * 16
STAIRCASE = (5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-capacities", type=int, nargs="+", default=(8, 16, 64))
    parser.add_argument("--launch-cycles", type=float, default=10.0)
    parser.add_argument("--scatter-cycles-per-row", type=float, default=0.1)
    args = parser.parse_args()

    schedules = {"uniform": UNIFORM, "staircase": STAIRCASE}
    rows: list[dict[str, float | int]] = []
    for capacity in args.batch_capacities:
        for staircase_fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            staircase_count = round(capacity * staircase_fraction)
            labels = ("staircase",) * staircase_count + ("uniform",) * (
                capacity - staircase_count
            )
            summary = analyze_schedule_queue(
                labels,
                schedules,
                batch_capacity=capacity,
                draft_layers=5,
                mlp_macs_per_layer=1,
                compute_macs_per_cycle=1,
                launch_cycles=args.launch_cycles,
                scatter_cycles_per_row=args.scatter_cycles_per_row,
                policy="arrival",
            )
            rows.append(
                {
                    "batch_capacity": capacity,
                    "staircase_fraction": staircase_fraction,
                    "grouped_over_dense": summary.total_grouped_cycles
                    / summary.total_dense_cycles,
                    "grouped_cycles": summary.total_grouped_cycles,
                    "dense_cycles": summary.total_dense_cycles,
                    "active_row_fraction": summary.active_row_fraction,
                    "mean_schedule_groups": summary.mean_schedule_groups,
                }
            )

    payload = {
        "kind": "reconfigurable_fabric_sweep",
        "warning": "normalized queue cost; not acceptance or end-to-end speedup",
        "parameters": {
            "launch_cycles": args.launch_cycles,
            "scatter_cycles_per_row": args.scatter_cycles_per_row,
            "draft_layers": 5,
            "block_size": 16,
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()

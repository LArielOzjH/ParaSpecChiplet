#!/usr/bin/env python3
"""Run a descriptive schedule-queue analysis on an acceptance trace.

The trace supplies only observed entering states. It does not prove that a
different schedule would produce the same acceptance, so this tool reports
queue/cost sensitivity rather than adaptive-policy speedup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paraspec.queue_model import analyze_schedule_queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--schedules", type=Path, required=True)
    parser.add_argument("--low-state-schedule", default="uniform")
    parser.add_argument("--high-state-schedule", default="protected8_staircase")
    parser.add_argument("--state-threshold", type=int, default=1)
    parser.add_argument("--batch-capacities", type=int, nargs="+", default=(8, 16, 32, 64))
    parser.add_argument("--mlp-macs-per-layer", type=int, default=1)
    parser.add_argument("--compute-macs-per-cycle", type=int, default=1)
    parser.add_argument("--launch-cycles", type=float, default=0.0)
    parser.add_argument("--scatter-cycles-per-row", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedules = json.loads(args.schedules.read_text())
    for name in (args.low_state_schedule, args.high_state_schedule):
        if name not in schedules:
            raise ValueError(f"schedule {name!r} is missing from schedule file")

    labels: list[str] = []
    previous_by_request: dict[str, int] = {}
    for line in args.trace.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("kind") != "dflash_verify":
            continue
        request_id = str(event.get("request_id", "default"))
        previous = previous_by_request.get(request_id, 0)
        labels.append(
            args.high_state_schedule
            if previous >= args.state_threshold
            else args.low_state_schedule
        )
        previous_by_request[request_id] = int(event["accepted_prefix"])
    if not labels:
        raise ValueError("trace contains no dflash_verify events")

    draft_layers = max(max(int(depth) for depth in values) for values in schedules.values())
    results = []
    for capacity in args.batch_capacities:
        for policy in ("arrival", "coalesced"):
            summary = analyze_schedule_queue(
                labels,
                schedules,
                batch_capacity=capacity,
                draft_layers=draft_layers,
                mlp_macs_per_layer=args.mlp_macs_per_layer,
                compute_macs_per_cycle=args.compute_macs_per_cycle,
                launch_cycles=args.launch_cycles,
                scatter_cycles_per_row=args.scatter_cycles_per_row,
                policy=policy,
            )
            results.append({"batch_capacity": capacity, **summary.__dict__})

    payload = {
        "events": len(labels),
        "state_threshold": args.state_threshold,
        "high_state_fraction": labels.count(args.high_state_schedule) / len(labels),
        "warning": "descriptive queue sensitivity; not adaptive acceptance or speedup evidence",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"events": len(labels), "results": len(results)}, indent=2))


if __name__ == "__main__":
    main()

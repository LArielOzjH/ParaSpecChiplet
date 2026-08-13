#!/usr/bin/env python3
"""Combine official width-acceptance traces with measured MLP latency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paraspec.fidelity_frontier import schedule_frontier


def trace_mean(path: Path, experiment: str) -> float:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    values = [row["accepted_prefix"] for row in rows if row["experiment"] == experiment]
    if not values:
        raise ValueError(f"no experiment {experiment!r} in {path}")
    return mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latency-data", type=Path, required=True)
    parser.add_argument("--uniform-trace", type=Path, required=True)
    parser.add_argument("--layer2-trace", type=Path, required=True)
    parser.add_argument("--layer23-trace", type=Path, required=True)
    parser.add_argument("--min-accepted-prefix", type=float, default=1.4)
    args = parser.parse_args()

    latency = json.loads(args.latency_data.read_text())
    rows = [row for row in latency["records"] if row["active_per_request"] == 16]
    latency_by_width = {
        float(row["width_fraction"]): float(row["reduced_ms"])
        for row in rows
        if row["width_fraction"] in (0.5, 1.0)
    }
    schedules = {
        "uniform": (trace_mean(args.uniform_trace, "uniform"), (1.0,) * 5),
        "layer2_half": (
            trace_mean(args.layer2_trace, "width_2_0.5"),
            (1.0, 1.0, 0.5, 1.0, 1.0),
        ),
        "layer23_half": (
            trace_mean(args.layer23_trace, "width_2_0.5_3_0.5"),
            (1.0, 1.0, 0.5, 0.5, 1.0),
        ),
    }
    frontier = schedule_frontier(
        schedules,
        mlp_latency_by_width=latency_by_width,
        attention_latency=0.0,
        min_accepted_prefix=args.min_accepted_prefix,
    )
    payload = {
        "kind": "official_fidelity_frontier",
        "warning": "acceptance traces plus single-layer MLP latency calibration; not end-to-end speedup",
        "min_accepted_prefix": args.min_accepted_prefix,
        "mlp_latency_by_width_ms": latency_by_width,
        "frontier": [point.__dict__ for point in frontier],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"frontier_points": len(frontier), "output": str(args.output)}))


if __name__ == "__main__":
    main()

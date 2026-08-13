#!/usr/bin/env python3
"""Compose official schedule choices from measured width/row latency data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paraspec.calibrated_latency import estimate_schedule_latency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-rows", type=int, default=16)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--attention-latency", type=float, default=0.0)
    args = parser.parse_args()

    payload = json.loads(args.latency_data.read_text())
    table = {
        (int(row["active_per_request"]), float(row["width_fraction"])): float(row["reduced_ms"])
        for row in payload["records"]
    }
    schedules = {
        "uniform": (1.0,) * args.layers,
        "layer2_half": tuple(0.5 if index == 2 else 1.0 for index in range(args.layers)),
        "layer23_half": tuple(0.5 if index in (2, 3) else 1.0 for index in range(args.layers)),
    }
    results = []
    for name, widths in schedules.items():
        result = estimate_schedule_latency(
            width_by_layer=widths,
            active_rows_by_layer=(args.active_rows,) * args.layers,
            latency_table=table,
            dense_attention_latency=args.attention_latency,
        )
        results.append(
            {
                "name": name,
                "width_by_layer": list(widths),
                "mlp_latency_ms": result.mlp_latency,
                "total_latency_ms": result.total_latency,
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "kind": "calibrated_schedule_latency",
                "warning": "composed fixed-batch MLP calibration; not end-to-end serving speedup",
                "batch_size": payload.get("batch_size"),
                "active_rows": args.active_rows,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"schedules": len(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()

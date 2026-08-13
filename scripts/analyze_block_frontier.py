#!/usr/bin/env python3
"""Build an empirical joint block-fidelity frontier from an official trace."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paraspec.block_frontier import empirical_block_frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--draft-layers", type=int, required=True)
    parser.add_argument("--min-survival", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observations: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for line in args.trace.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("ablation_mode") != "mlp":
            continue
        group = tuple(int(index) for index in record["ablated_layers"])
        observations[group].append(float(record["accepted_prefix"]))
    if not observations:
        raise ValueError("trace contains no MLP ablation records")

    means = {
        group: sum(values) / len(values) for group, values in observations.items()
    }
    frontier = empirical_block_frontier(
        means,
        draft_layers=args.draft_layers,
        min_survival=args.min_survival,
    )
    payload = {
        "kind": "empirical_dflash_block_frontier",
        "trace": str(args.trace),
        "draft_layers": args.draft_layers,
        "min_survival": args.min_survival,
        "warning": "nominal_mlp_work is analytical; this is not a speedup result",
        "observed_groups": [
            {
                "bypassed_layers": list(group),
                "mean_accepted_prefix": means[group],
                "events": len(observations[group]),
            }
            for group in sorted(means)
        ],
        "frontier": [
            {
                "bypassed_layers": list(point.bypassed_layers),
                "mean_accepted_prefix": point.mean_accepted_prefix,
                "nominal_mlp_work": point.nominal_mlp_work,
            }
            for point in frontier
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"observed_groups": len(means), "frontier_points": len(frontier)}))


if __name__ == "__main__":
    main()

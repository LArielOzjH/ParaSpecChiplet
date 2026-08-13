#!/usr/bin/env python3
"""Replay schedule selection on a saved DFlash acceptance trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paraspec.controller import ScheduleOption, choose_schedule
from paraspec.trace_io import load_acceptance_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--protected-prefix", type=int, required=True)
    parser.add_argument("--max-prefix-drop", type=float, default=0.02)
    args = parser.parse_args()

    trace = load_acceptance_trace(args.trace)
    payload = json.loads(args.options.read_text())
    options = tuple(
        ScheduleOption(
            name=str(option["name"]),
            depth_by_position=tuple(int(value) for value in option["depth_by_position"]),
            predicted_survival=tuple(float(value) for value in option["predicted_survival"]),
            cost=float(option["cost"]),
        )
        for option in payload
    )
    decision = choose_schedule(
        trace,
        options,
        protected_prefix=args.protected_prefix,
        max_prefix_drop=args.max_prefix_drop,
    )
    print(
        json.dumps(
            {
                "name": decision.name,
                "depth_by_position": decision.depth_by_position,
                "score": decision.score,
                "rejected": decision.rejected,
                "history_cycles": len(trace.acceptance_lengths),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

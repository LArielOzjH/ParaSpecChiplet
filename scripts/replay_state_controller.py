#!/usr/bin/env python3
"""Replay state-specific schedule selection on an official DFlash trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paraspec.controller import ScheduleOption, choose_schedule_by_state
from paraspec.state_oracle import StateTrace


def load_state_trace(path: Path, bucket_size: int) -> StateTrace:
    lengths: list[int] = []
    states: list[str] = []
    block_size: int | None = None
    previous_by_request: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("kind") != "dflash_verify":
            continue
        current_size = int(event["block_size"])
        if block_size is None:
            block_size = current_size
        elif block_size != current_size:
            raise ValueError("all events must share block_size")
        request_id = str(event.get("request_id", "default"))
        previous = previous_by_request.get(request_id, 0)
        states.append(f"prev_prefix_bucket_{previous // bucket_size}")
        lengths.append(int(event["accepted_prefix"]))
        previous_by_request[request_id] = int(event["accepted_prefix"])
    if block_size is None:
        raise ValueError("no dflash_verify events found")
    return StateTrace.from_sequences(
        acceptance_lengths=lengths,
        state_by_cycle=states,
        block_size=block_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket-size", type=int, default=1)
    parser.add_argument("--protected-prefix", type=int, required=True)
    parser.add_argument("--max-prefix-drop", type=float, default=0.02)
    args = parser.parse_args()
    if args.bucket_size <= 0:
        raise ValueError("bucket size must be positive")

    trace = load_state_trace(args.trace, args.bucket_size)
    raw_options = json.loads(args.options.read_text())
    options_by_state = {
        str(state): tuple(
            ScheduleOption(
                name=str(option["name"]),
                depth_by_position=tuple(int(value) for value in option["depth_by_position"]),
                predicted_survival=tuple(float(value) for value in option["predicted_survival"]),
                cost=float(option["cost"]),
            )
            for option in options
        )
        for state, options in raw_options.items()
    }
    decisions = choose_schedule_by_state(
        trace,
        options_by_state,
        protected_prefix=args.protected_prefix,
        max_prefix_drop=args.max_prefix_drop,
    )
    payload = {
        state: {
            "name": decision.name,
            "depth_by_position": decision.depth_by_position,
            "score": decision.score,
            "rejected": decision.rejected,
        }
        for state, decision in decisions.items()
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"states": len(payload), "cycles": len(trace.acceptance_lengths)}, indent=2))


if __name__ == "__main__":
    main()

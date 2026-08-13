#!/usr/bin/env python3
"""Analyze block survival conditioned on the state entering each block."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from paraspec.state_oracle import (
    StateTrace,
    conditional_prefix_survival,
    expected_committed_value,
    transition_counts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=1,
        help="Quantize previous accepted-prefix state; 1 keeps exact states.",
    )
    args = parser.parse_args()
    if args.bucket_size <= 0:
        raise ValueError("bucket size must be positive")

    lengths: list[int] = []
    states: list[str] = []
    block_size: int | None = None
    previous_by_request: dict[str, int] = {}
    for line_number, line in enumerate(args.trace.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if event.get("kind") != "dflash_verify":
            continue
        current_block_size = int(event["block_size"])
        if block_size is None:
            block_size = current_block_size
        elif block_size != current_block_size:
            raise ValueError("all events must share block_size")
        request_id = str(event.get("request_id", "default"))
        previous = previous_by_request.get(request_id, 0)
        state = previous // args.bucket_size
        lengths.append(int(event["accepted_prefix"]))
        states.append(f"prev_prefix_bucket_{state}")
        previous_by_request[request_id] = int(event["accepted_prefix"])

    if block_size is None:
        raise ValueError("no dflash_verify events found")
    trace = StateTrace.from_sequences(
        acceptance_lengths=lengths,
        state_by_cycle=states,
        block_size=block_size,
    )
    conditional = conditional_prefix_survival(trace)
    payload = {
        "block_size": block_size,
        "bucket_size": args.bucket_size,
        "cycles": len(lengths),
        "conditional_prefix_survival": conditional,
        "expected_committed_value": expected_committed_value(trace),
        "transition_counts": {
            f"{state}|accepted_{accepted}": count
            for (state, accepted), count in transition_counts(trace).items()
        },
        "interpretation_warning": "state conditioning is descriptive; it does not establish causality or predict selective-depth acceptance",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"cycles": len(lengths), "states": len(conditional)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a deliberately non-authoritative position-accuracy proxy sweep.

This is not a replacement for prefix-survival traces. It only turns the
public validation accuracy fixture into a transparent sanity check for the
schedule oracle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paraspec.public_fixture import load_position_accuracy
from paraspec.schedule import schedule_efficiency, staircase_depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--full-depth", type=int, default=3)
    parser.add_argument("--min-depth", type=int, default=1)
    args = parser.parse_args()

    fixture = load_position_accuracy(args.fixture)
    uniform_depth = (args.full_depth,) * len(fixture.accuracy_by_position)
    staircase = staircase_depth(
        len(fixture.accuracy_by_position), args.full_depth, args.min_depth
    )
    uniform = schedule_efficiency(fixture.accuracy_by_position, uniform_depth, 1.0)
    tapered = schedule_efficiency(fixture.accuracy_by_position, staircase, 1.0)
    output = {
        "model": fixture.model,
        "source": fixture.source,
        "proxy_warning": "marginal position accuracy is not prefix survival",
        "position_accuracy": fixture.accuracy_by_position,
        "tail_to_head_ratio": fixture.accuracy_by_position[-1]
        / fixture.accuracy_by_position[0],
        "uniform": uniform.__dict__,
        "staircase": {"depth": staircase, **tapered.__dict__},
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()


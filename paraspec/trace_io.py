from __future__ import annotations

import json
from pathlib import Path

from .trace_oracle import Trace


def load_acceptance_trace(path: str | Path) -> Trace:
    """Load DFlash verification prefix lengths from a JSONL event capture."""

    block_size: int | None = None
    acceptance_lengths: list[int] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if event.get("kind") != "dflash_verify":
            continue
        event_block_size = int(event["block_size"])
        if block_size is None:
            block_size = event_block_size
        elif block_size != event_block_size:
            raise ValueError("all DFlash verification events must share block_size")
        acceptance_lengths.append(int(event["accepted_prefix"]))
    if block_size is None or not acceptance_lengths:
        raise ValueError("no dflash_verify events found")
    return Trace.from_acceptance_lengths(acceptance_lengths, block_size)


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PositionAccuracy:
    model: str
    source: str
    accuracy_by_position: tuple[float, ...]


def load_position_accuracy(path: str | Path) -> PositionAccuracy:
    payload = json.loads(Path(path).read_text())
    positions = sorted(
        (
            int(key.removeprefix("position_").removesuffix("_acc_epoch")),
            float(value),
        )
        for key, value in payload.items()
        if key.startswith("position_") and key.endswith("_acc_epoch")
    )
    if not positions:
        raise ValueError("fixture contains no position accuracy fields")
    values = tuple(value for _, value in positions)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("position accuracies must be in [0, 1]")
    return PositionAccuracy(
        model=str(payload["model"]),
        source=str(payload["source"]),
        accuracy_by_position=values,
    )


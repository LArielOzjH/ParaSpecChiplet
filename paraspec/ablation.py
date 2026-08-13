from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LayerPositionTrace:
    block_size: int
    draft_layers: int
    baseline_prefix_survival: tuple[float, ...]
    survival_after_layer: tuple[tuple[float, ...], ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "LayerPositionTrace":
        block_size = int(payload["block_size"])
        draft_layers = int(payload["draft_layers"])
        baseline = tuple(float(x) for x in payload["baseline_prefix_survival"])  # type: ignore[arg-type]
        after = tuple(
            tuple(float(x) for x in row) for row in payload["survival_after_layer"]  # type: ignore[arg-type]
        )
        if block_size <= 0 or draft_layers <= 0:
            raise ValueError("block_size and draft_layers must be positive")
        if len(baseline) != block_size:
            raise ValueError("baseline must have one value per position")
        if len(after) != draft_layers or any(len(row) != block_size for row in after):
            raise ValueError("survival_after_layer must be draft_layers x block_size")
        values = baseline + tuple(value for row in after for value in row)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("survival values must be in [0, 1]")
        return cls(block_size, draft_layers, baseline, after)


def marginal_layer_value(
    trace: LayerPositionTrace,
) -> tuple[tuple[float, ...], ...]:
    """Return per-position survival gain from adding each draft layer.

    Layer zero is measured against the no-layer baseline; later layers are
    measured against the immediately preceding checkpoint. Negative values
    are retained because an approximation or ablation can hurt survival.
    """

    previous: Sequence[float] = trace.baseline_prefix_survival
    gains: list[tuple[float, ...]] = []
    for checkpoint in trace.survival_after_layer:
        gains.append(tuple(value - old for value, old in zip(checkpoint, previous)))
        previous = checkpoint
    return tuple(gains)


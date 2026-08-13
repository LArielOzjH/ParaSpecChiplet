"""Acceptance-calibrated frontier for block-level MLP fidelity schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FidelitySchedulePoint:
    name: str
    mean_accepted_prefix: float
    width_by_layer: tuple[float, ...]
    total_latency: float


def schedule_frontier(
    schedules: Mapping[str, tuple[float, Sequence[float]]],
    *,
    mlp_latency_by_width: Mapping[float, float],
    attention_latency: float,
    min_accepted_prefix: float,
) -> tuple[FidelitySchedulePoint, ...]:
    """Return safe, non-dominated schedules using measured width latencies.

    The latency model assumes dense attention and one MLP latency lookup per
    draft layer. It is a calibrated schedule proxy, not end-to-end timing.
    """

    if min_accepted_prefix <= 0:
        raise ValueError("min_accepted_prefix must be positive")
    if attention_latency < 0 or any(value < 0 for value in mlp_latency_by_width.values()):
        raise ValueError("latencies must be non-negative")
    lookup = {float(width): float(latency) for width, latency in mlp_latency_by_width.items()}
    points: list[FidelitySchedulePoint] = []
    for name, (accepted, widths) in schedules.items():
        width_tuple = tuple(float(width) for width in widths)
        if not width_tuple:
            raise ValueError("width_by_layer must not be empty")
        if any(width not in lookup for width in width_tuple):
            raise ValueError("schedule width has no measured MLP latency")
        accepted_value = float(accepted)
        if accepted_value < min_accepted_prefix:
            continue
        points.append(
            FidelitySchedulePoint(
                name=str(name),
                mean_accepted_prefix=accepted_value,
                width_by_layer=width_tuple,
                total_latency=attention_latency + sum(lookup[width] for width in width_tuple),
            )
        )
    frontier = [
        point
        for point in points
        if not any(
            other.mean_accepted_prefix >= point.mean_accepted_prefix
            and other.total_latency <= point.total_latency
            and (
                other.mean_accepted_prefix > point.mean_accepted_prefix
                or other.total_latency < point.total_latency
            )
            for other in points
        )
    ]
    return tuple(sorted(frontier, key=lambda point: (point.total_latency, point.name)))

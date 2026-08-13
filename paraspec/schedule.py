from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ScheduleEfficiency:
    expected_committed_value: float
    compute_cost: float
    value_per_cost: float


def staircase_depth(block_size: int, full_depth: int, min_depth: int) -> tuple[int, ...]:
    """Return a monotone depth schedule from full prefix to cheaper tail."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if min_depth <= 0 or full_depth < min_depth:
        raise ValueError("require 0 < min_depth <= full_depth")
    if block_size == 1:
        return (full_depth,)
    step = (full_depth - min_depth) / (block_size - 1)
    depths = tuple(round(full_depth - index * step) for index in range(block_size))
    return tuple(max(min_depth, min(full_depth, depth)) for depth in depths)


def schedule_efficiency(
    survival: Sequence[float],
    depth_by_position: Sequence[int],
    cost_per_layer: float,
) -> ScheduleEfficiency:
    """Score a schedule using prefix survival as the committed-value proxy."""

    if len(survival) != len(depth_by_position):
        raise ValueError("survival and depth schedule must have equal length")
    if cost_per_layer <= 0:
        raise ValueError("cost_per_layer must be positive")
    if any(not 0.0 <= value <= 1.0 for value in survival):
        raise ValueError("survival values must be in [0, 1]")
    if any(depth <= 0 for depth in depth_by_position):
        raise ValueError("depths must be positive")
    expected_value = float(sum(survival))
    compute_cost = float(sum(depth_by_position) * cost_per_layer)
    return ScheduleEfficiency(expected_value, compute_cost, expected_value / compute_cost)


"""Offline schedule-space and Pareto analysis for heterogeneous DFlash."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from .ablation import LayerPositionTrace, marginal_layer_value
from .chiplet_cost import ChipletCost, estimate_chiplet_cost


@dataclass(frozen=True)
class FrontierPoint:
    depth_by_position: tuple[int, ...]
    predicted_survival: tuple[float, ...]
    predicted_value: float
    cost: ChipletCost


def enumerate_depth_schedules(
    *, block_size: int, draft_layers: int, min_depth: int, protected_prefix: int
) -> tuple[tuple[int, ...], ...]:
    """Enumerate non-increasing depth vectors with a full protected prefix."""

    if block_size <= 0 or draft_layers <= 0:
        raise ValueError("block_size and draft_layers must be positive")
    if not 1 <= min_depth <= draft_layers:
        raise ValueError("min_depth must be within the draft depth")
    if not 0 <= protected_prefix <= block_size:
        raise ValueError("protected_prefix must be within the block")
    schedules = []
    for suffix in product(range(min_depth, draft_layers + 1), repeat=block_size - protected_prefix):
        depths = (draft_layers,) * protected_prefix + suffix
        if all(left >= right for left, right in zip(depths, depths[1:])):
            schedules.append(tuple(depths))
    return tuple(sorted(schedules))


def _predict_survival(trace: LayerPositionTrace, depths: Sequence[int]) -> tuple[float, ...]:
    """Compose measured layer gains additively as an explicitly approximate proxy."""

    gains = marginal_layer_value(trace)
    predicted = list(trace.baseline_prefix_survival)
    for position, depth in enumerate(depths):
        predicted[position] += sum(gain[position] for gain in gains[:depth])
    return tuple(max(0.0, min(1.0, value)) for value in predicted)


def pareto_frontier(
    trace: LayerPositionTrace,
    *,
    depth_schedules: Sequence[Sequence[int]],
    cost_kwargs: Mapping[str, object],
) -> tuple[FrontierPoint, ...]:
    """Return points not dominated in predicted value and total cost.

    The prediction is additive and therefore not a substitute for selective
    execution. It is useful for ranking candidates before expensive real runs.
    """

    points: list[FrontierPoint] = []
    for schedule in depth_schedules:
        depths = tuple(int(depth) for depth in schedule)
        if len(depths) != trace.block_size:
            raise ValueError("depth schedule must match trace block size")
        if any(depth < 1 or depth > trace.draft_layers for depth in depths):
            raise ValueError("depths must be within the trace draft depth")
        predicted = _predict_survival(trace, depths)
        cost = estimate_chiplet_cost(depths, **dict(cost_kwargs))
        points.append(FrontierPoint(depths, predicted, sum(predicted), cost))

    frontier = []
    for point in points:
        dominated = any(
            other.predicted_value >= point.predicted_value
            and other.cost.total_cycles <= point.cost.total_cycles
            and (
                other.predicted_value > point.predicted_value
                or other.cost.total_cycles < point.cost.total_cycles
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return tuple(sorted(frontier, key=lambda item: item.cost.total_cycles))

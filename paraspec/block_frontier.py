"""Empirical Pareto analysis for joint draft-block fidelity schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .block_ablation import validate_layer_indices


@dataclass(frozen=True)
class BlockFrontierPoint:
    """One measured or replayed joint block-fidelity schedule."""

    bypassed_layers: tuple[int, ...]
    mean_accepted_prefix: float
    nominal_mlp_work: int


def empirical_block_frontier(
    mean_prefix_by_group: Mapping[Sequence[int], float],
    *,
    draft_layers: int,
    min_survival: float,
) -> tuple[BlockFrontierPoint, ...]:
    """Return safe, non-dominated schedules from measured group means.

    ``mean_prefix_by_group`` maps bypassed zero-based draft layer groups to
    observed mean accepted-prefix length. ``nominal_mlp_work`` counts draft
    MLP blocks remaining and is an analytical work proxy, not a timing result.
    A schedule is dominated when another schedule has at least as much
    acceptance and no more nominal work, with one strict improvement.
    """

    if min_survival <= 0:
        raise ValueError("min_survival must be positive")
    points: list[BlockFrontierPoint] = []
    seen: set[tuple[int, ...]] = set()
    for raw_group, raw_mean in mean_prefix_by_group.items():
        group = validate_layer_indices(raw_group, draft_layers=draft_layers)
        if group in seen:
            continue
        seen.add(group)
        mean_prefix = float(raw_mean)
        if mean_prefix < min_survival:
            continue
        points.append(
            BlockFrontierPoint(
                bypassed_layers=group,
                mean_accepted_prefix=mean_prefix,
                nominal_mlp_work=draft_layers - len(group),
            )
        )

    frontier = [
        point
        for point in points
        if not any(
            other.mean_accepted_prefix >= point.mean_accepted_prefix
            and other.nominal_mlp_work <= point.nominal_mlp_work
            and (
                other.mean_accepted_prefix > point.mean_accepted_prefix
                or other.nominal_mlp_work < point.nominal_mlp_work
            )
            for other in points
        )
    ]
    return tuple(
        sorted(frontier, key=lambda point: (point.nominal_mlp_work, point.bypassed_layers))
    )

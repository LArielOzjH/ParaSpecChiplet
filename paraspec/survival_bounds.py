"""Conservative prefix-survival bounds from marginal position accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SurvivalBounds:
    lower: tuple[float, ...]
    upper: tuple[float, ...]


def prefix_survival_bounds(
    accuracy_by_position: Sequence[float], *, full_block_accuracy: float | None = None
) -> SurvivalBounds:
    """Return Fréchet bounds for ``P(all positions <= i are correct)``.

    Marginal accuracies alone do not identify prefix survival. For prefix ``i``,
    the lower bound is ``max(0, sum(p[:i]) - (i-1))`` and the upper bound is
    ``min(p[:i])``. If a trustworthy full-block accuracy is supplied, it is a
    lower bound for every prefix as well.
    """

    values = tuple(float(value) for value in accuracy_by_position)
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("accuracy values must be a non-empty sequence in [0, 1]")
    if full_block_accuracy is not None and not 0.0 <= full_block_accuracy <= 1.0:
        raise ValueError("full-block accuracy must be in [0, 1]")
    if full_block_accuracy is not None and full_block_accuracy > min(values):
        raise ValueError(
            "full-block accuracy cannot exceed a marginal position accuracy"
        )
    lower = []
    upper = []
    for index in range(1, len(values) + 1):
        lower_bound = max(0.0, sum(values[:index]) - (index - 1))
        if full_block_accuracy is not None:
            lower_bound = max(lower_bound, float(full_block_accuracy))
        lower.append(lower_bound)
        upper.append(min(values[:index]))
    return SurvivalBounds(tuple(lower), tuple(upper))

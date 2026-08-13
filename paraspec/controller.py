"""Replay oracle for survival-aware schedule selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .trace_oracle import Trace, prefix_survival


@dataclass(frozen=True)
class ScheduleOption:
    """A schedule plus held-out survival and cost estimates."""

    name: str
    depth_by_position: tuple[int, ...]
    predicted_survival: tuple[float, ...]
    cost: float


@dataclass(frozen=True)
class ScheduleDecision:
    name: str
    depth_by_position: tuple[int, ...]
    score: float
    rejected: tuple[str, ...]


def choose_schedule(
    history: Trace,
    options: Sequence[ScheduleOption],
    *,
    protected_prefix: int,
    max_prefix_drop: float,
) -> ScheduleDecision:
    """Select the best safe option using recent prefix-survival observations.

    ``predicted_survival`` must be produced by an offline/held-out evaluator
    for the current request state. This function deliberately does not invent
    an acceptance model; it only applies a conservative protected-prefix gate
    and scores committed-value per supplied cost.
    """

    if not options:
        raise ValueError("at least one schedule option is required")
    if not 1 <= protected_prefix <= history.block_size:
        raise ValueError("protected_prefix must be within the block")
    if not 0.0 <= max_prefix_drop <= 1.0:
        raise ValueError("max_prefix_drop must be in [0, 1]")
    observed = prefix_survival(history)
    safe: list[tuple[ScheduleOption, float]] = []
    rejected: list[str] = []
    for option in options:
        if len(option.depth_by_position) != history.block_size:
            raise ValueError("schedule depth must match history block size")
        if len(option.predicted_survival) != history.block_size:
            raise ValueError("predicted survival must match history block size")
        if any(depth <= 0 for depth in option.depth_by_position):
            raise ValueError("schedule depths must be positive")
        if any(not 0.0 <= value <= 1.0 for value in option.predicted_survival):
            raise ValueError("predicted survival values must be in [0, 1]")
        if option.cost <= 0:
            raise ValueError("schedule cost must be positive")
        protected_ok = all(
            predicted + max_prefix_drop >= actual
            for predicted, actual in zip(
                option.predicted_survival[:protected_prefix],
                observed[:protected_prefix],
            )
        )
        if not protected_ok:
            rejected.append(option.name)
            continue
        score = sum(option.predicted_survival) / option.cost
        safe.append((option, score))
    if not safe:
        raise ValueError("no schedule satisfies the protected-prefix constraint")
    selected, score = max(safe, key=lambda item: item[1])
    return ScheduleDecision(
        name=selected.name,
        depth_by_position=selected.depth_by_position,
        score=float(score),
        rejected=tuple(rejected),
    )

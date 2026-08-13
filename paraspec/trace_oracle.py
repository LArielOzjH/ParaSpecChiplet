from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Trace:
    """Acceptance-prefix observations from repeated DFlash verification cycles."""

    acceptance_lengths: tuple[int, ...]
    block_size: int

    @classmethod
    def from_acceptance_lengths(
        cls, acceptance_lengths: Iterable[int], block_size: int
    ) -> "Trace":
        lengths = tuple(int(length) for length in acceptance_lengths)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not lengths:
            raise ValueError("at least one acceptance length is required")
        if any(length < 0 or length > block_size for length in lengths):
            raise ValueError("acceptance lengths must be in [0, block_size]")
        return cls(lengths, int(block_size))


def prefix_survival(trace: Trace) -> tuple[float, ...]:
    """Return ``P(A >= i)`` for positions ``i = 1..block_size``."""

    count = len(trace.acceptance_lengths)
    return tuple(
        sum(length >= position for length in trace.acceptance_lengths) / count
        for position in range(1, trace.block_size + 1)
    )


def conditional_hazard(trace: Trace) -> tuple[float, ...]:
    """Return ``P(A=i-1 | A>=i-1)`` for each position ``i``.

    The first value is the chance that the first speculative position fails.
    A zero denominator is reported as zero because that position is never
    reached in the observed trace.
    """

    hazards: list[float] = []
    total = len(trace.acceptance_lengths)
    for position in range(1, trace.block_size + 1):
        survivors = sum(length >= position - 1 for length in trace.acceptance_lengths)
        failures = sum(length == position - 1 for length in trace.acceptance_lengths)
        hazards.append(failures / survivors if survivors else 0.0)
    return tuple(hazards)


def value_density(
    survival: Sequence[float], extra_cost: Sequence[float]
) -> tuple[float, ...]:
    """Normalize each position's survival value by its incremental cost."""

    if len(survival) != len(extra_cost):
        raise ValueError("survival and extra_cost must have equal length")
    if any(cost <= 0 for cost in extra_cost):
        raise ValueError("extra costs must be positive")
    return tuple(value / cost for value, cost in zip(survival, extra_cost))


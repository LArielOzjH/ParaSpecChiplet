"""State-conditioned survival statistics for sequential DFlash blocks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence


@dataclass(frozen=True)
class StateTrace:
    """Acceptance observations paired with the state entering each block."""

    acceptance_lengths: tuple[int, ...]
    state_by_cycle: tuple[Hashable, ...]
    block_size: int

    @classmethod
    def from_sequences(
        cls,
        *,
        acceptance_lengths: Iterable[int],
        state_by_cycle: Iterable[Hashable],
        block_size: int,
    ) -> "StateTrace":
        lengths = tuple(int(value) for value in acceptance_lengths)
        states = tuple(state_by_cycle)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not lengths:
            raise ValueError("at least one acceptance length is required")
        if len(lengths) != len(states):
            raise ValueError("acceptance lengths and states must have the same length")
        if any(length < 0 or length > block_size for length in lengths):
            raise ValueError("acceptance lengths must be in [0, block_size]")
        return cls(lengths, states, int(block_size))


def conditional_prefix_survival(trace: StateTrace) -> dict[Hashable, tuple[float, ...]]:
    """Return ``P(A >= i | entering-state)`` for every observed state."""

    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for state, length in zip(trace.state_by_cycle, trace.acceptance_lengths):
        grouped[state].append(length)
    return {
        state: tuple(
            sum(length >= position for length in lengths) / len(lengths)
            for position in range(1, trace.block_size + 1)
        )
        for state, lengths in grouped.items()
    }


def expected_committed_value(trace: StateTrace) -> dict[Hashable, float]:
    """Return expected accepted draft-prefix length conditioned on state."""

    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for state, length in zip(trace.state_by_cycle, trace.acceptance_lengths):
        grouped[state].append(length)
    return {state: sum(lengths) / len(lengths) for state, lengths in grouped.items()}


def transition_counts(trace: StateTrace) -> dict[tuple[Hashable, int], int]:
    """Count entering-state to observed accepted-prefix transitions."""

    return dict(Counter(zip(trace.state_by_cycle, trace.acceptance_lengths)))

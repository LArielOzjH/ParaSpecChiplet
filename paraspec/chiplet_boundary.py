"""Break-even analysis for equal-resource chiplet specialization."""

from __future__ import annotations


def chiplet_break_even_parallel_gain(
    *,
    compute_cycles: float,
    overhead_cycles: float,
) -> float:
    """Return the minimum compute gain needed to amortize chiplet overhead.

    The equal-resource monolithic reference takes ``compute_cycles``. A
    chiplet realization with the same aggregate compute capacity and explicit
    link/router/synchronization overhead takes
    ``compute_cycles / parallel_gain + overhead_cycles``. Equality gives the
    returned break-even gain. Values below it lose to monolithic execution.
    """

    if compute_cycles <= 0:
        raise ValueError("compute_cycles must be positive")
    if overhead_cycles < 0:
        raise ValueError("overhead_cycles must be non-negative")
    return float((compute_cycles + overhead_cycles) / compute_cycles)

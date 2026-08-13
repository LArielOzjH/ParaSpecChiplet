from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ChipletCost:
    compute_cycles: float
    link_cycles: float
    synchronization_cycles: float
    total_cycles: float


def estimate_chiplet_cost(
    depth_by_position: Sequence[int],
    macs_per_layer: int,
    compute_macs_per_cycle: int,
    activation_bytes_per_position: int,
    link_bytes_per_cycle: int,
    synchronization_cycles: int,
) -> ChipletCost:
    """Estimate a transparent upper-bound cost for split draft execution.

    The model intentionally exposes rather than hides link and synchronization
    penalties. It assumes each active position crosses the chiplet boundary
    once; future versions can replace that term with measured multicast/reuse.
    """

    if not depth_by_position or any(depth <= 0 for depth in depth_by_position):
        raise ValueError("depth_by_position must contain positive depths")
    if min(
        macs_per_layer,
        compute_macs_per_cycle,
        activation_bytes_per_position,
        link_bytes_per_cycle,
    ) <= 0:
        raise ValueError("cost parameters must be positive")
    if synchronization_cycles < 0:
        raise ValueError("synchronization_cycles must be non-negative")
    compute_cycles = (
        sum(depth_by_position) * macs_per_layer / compute_macs_per_cycle
    )
    total_link_bytes = len(depth_by_position) * activation_bytes_per_position
    link_cycles = total_link_bytes / link_bytes_per_cycle
    total_cycles = compute_cycles + link_cycles + synchronization_cycles
    return ChipletCost(
        compute_cycles=float(compute_cycles),
        link_cycles=float(link_cycles),
        synchronization_cycles=float(synchronization_cycles),
        total_cycles=float(total_cycles),
    )


"""Chiplet versus equal-resource monolithic sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .chiplet_cost import estimate_chiplet_cost, estimate_monolithic_cost


@dataclass(frozen=True)
class FabricPoint:
    link_bytes_per_cycle: float
    multicast_reuse: float
    chiplet_total_cycles: float
    monolithic_total_cycles: float
    chiplet_speedup_over_monolithic: float


def sweep_chiplet_tradeoff(
    *,
    depth_by_position: Sequence[int],
    base_cost_kwargs: Mapping[str, object],
    link_bytes_per_cycle_values: Sequence[float],
    multicast_reuse_values: Sequence[float],
) -> tuple[FabricPoint, ...]:
    """Evaluate chiplet overhead sensitivity against a dense baseline."""

    monolithic = estimate_monolithic_cost(
        depth_by_position,
        macs_per_layer=int(base_cost_kwargs["macs_per_layer"]),
        compute_macs_per_cycle=int(base_cost_kwargs["compute_macs_per_cycle"]),
        synchronization_cycles=int(base_cost_kwargs.get("synchronization_cycles", 0)),
    )
    points = []
    for link_bandwidth in link_bytes_per_cycle_values:
        for reuse in multicast_reuse_values:
            kwargs = dict(base_cost_kwargs)
            kwargs["link_bytes_per_cycle"] = link_bandwidth
            kwargs["activation_multicast_reuse"] = reuse
            chiplet = estimate_chiplet_cost(depth_by_position, **kwargs)
            points.append(
                FabricPoint(
                    link_bytes_per_cycle=float(link_bandwidth),
                    multicast_reuse=float(reuse),
                    chiplet_total_cycles=chiplet.total_cycles,
                    monolithic_total_cycles=monolithic.total_cycles,
                    chiplet_speedup_over_monolithic=(
                        monolithic.total_cycles / chiplet.total_cycles
                    ),
                )
            )
    return tuple(points)

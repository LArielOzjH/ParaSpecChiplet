from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ChipletCost:
    compute_cycles: float
    link_cycles: float
    synchronization_cycles: float
    total_cycles: float
    router_cycles: float = 0.0


def estimate_monolithic_cost(
    depth_by_position: Sequence[int],
    macs_per_layer: int,
    compute_macs_per_cycle: int,
    synchronization_cycles: int = 0,
) -> ChipletCost:
    """Estimate equal-resource dense execution with idle lanes at max depth.

    This is a conservative monolithic baseline: the accelerator has one lane
    per block position and runs every lane through the deepest selected draft
    depth. Positions that need fewer layers are idle during those upper-layer
    cycles. It intentionally excludes chiplet link/router terms.
    """

    if not depth_by_position or any(depth <= 0 for depth in depth_by_position):
        raise ValueError("depth_by_position must contain positive depths")
    if macs_per_layer <= 0 or compute_macs_per_cycle <= 0:
        raise ValueError("compute parameters must be positive")
    if synchronization_cycles < 0:
        raise ValueError("synchronization_cycles must be non-negative")
    compute_cycles = (
        max(depth_by_position)
        * len(depth_by_position)
        * macs_per_layer
        / compute_macs_per_cycle
    )
    total_cycles = compute_cycles + synchronization_cycles
    return ChipletCost(
        compute_cycles=float(compute_cycles),
        link_cycles=0.0,
        synchronization_cycles=float(synchronization_cycles),
        total_cycles=float(total_cycles),
    )


def estimate_mlp_gated_cost(
    depth_by_position: Sequence[int],
    attention_macs_per_layer: int,
    mlp_macs_per_layer: int,
    compute_macs_per_cycle: int,
    synchronization_cycles: int = 0,
) -> ChipletCost:
    """Estimate dense attention plus position-gated upper-layer MLP work.

    Attention runs for every position at every layer up to the maximum depth,
    preserving bidirectional block context. MLP work is paid only for the
    position/layer pairs present in ``depth_by_position``.
    """

    if not depth_by_position or any(depth <= 0 for depth in depth_by_position):
        raise ValueError("depth_by_position must contain positive depths")
    if min(attention_macs_per_layer, mlp_macs_per_layer, compute_macs_per_cycle) <= 0:
        raise ValueError("compute parameters must be positive")
    if synchronization_cycles < 0:
        raise ValueError("synchronization_cycles must be non-negative")
    attention_work = max(depth_by_position) * len(depth_by_position) * attention_macs_per_layer
    mlp_work = sum(depth_by_position) * mlp_macs_per_layer
    compute_cycles = (attention_work + mlp_work) / compute_macs_per_cycle
    total_cycles = compute_cycles + synchronization_cycles
    return ChipletCost(
        compute_cycles=float(compute_cycles),
        link_cycles=0.0,
        synchronization_cycles=float(synchronization_cycles),
        total_cycles=float(total_cycles),
    )


def estimate_width_aware_mlp_cost(
    width_by_layer: Sequence[float],
    *,
    block_size: int,
    attention_macs_per_layer: int,
    mlp_macs_per_layer: int,
    compute_macs_per_cycle: int,
    synchronization_cycles: int = 0,
) -> ChipletCost:
    """Estimate dense-attention cost with per-layer MLP width fractions.

    This models one DFlash block with a fixed draft depth. Attention remains
    dense for every position and layer; only MLP MACs are scaled by the
    supplied width fractions. The result is an analytical work model, not a
    measured latency or acceptance claim.
    """

    widths = tuple(float(width) for width in width_by_layer)
    if not widths or any(not 0.0 < width <= 1.0 for width in widths):
        raise ValueError("width_by_layer values must be within (0, 1]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if min(attention_macs_per_layer, mlp_macs_per_layer, compute_macs_per_cycle) <= 0:
        raise ValueError("compute parameters must be positive")
    if synchronization_cycles < 0:
        raise ValueError("synchronization_cycles must be non-negative")
    attention_work = len(widths) * block_size * attention_macs_per_layer
    mlp_work = block_size * sum(widths) * mlp_macs_per_layer
    compute_cycles = (attention_work + mlp_work) / compute_macs_per_cycle
    total_cycles = compute_cycles + synchronization_cycles
    return ChipletCost(
        compute_cycles=float(compute_cycles),
        link_cycles=0.0,
        synchronization_cycles=float(synchronization_cycles),
        total_cycles=float(total_cycles),
    )


def estimate_chiplet_mlp_gated_cost(
    depth_by_position: Sequence[int],
    attention_macs_per_layer: int,
    mlp_macs_per_layer: int,
    compute_macs_per_cycle: int,
    activation_bytes_per_position: int,
    link_bytes_per_cycle: int,
    synchronization_cycles: int,
    activation_multicast_reuse: float = 1.0,
    router_cycles_per_position: float = 0.0,
) -> ChipletCost:
    """Model dense attention plus gated MLP with explicit chiplet traffic.

    Attention is paid for every block position through the maximum selected
    depth. MLP work follows the per-position depth vector. The link term models
    one activation transfer per position with configurable multicast reuse; it
    is intentionally conservative and does not assume free on-chip movement.
    """

    if not depth_by_position or any(depth <= 0 for depth in depth_by_position):
        raise ValueError("depth_by_position must contain positive depths")
    if min(
        attention_macs_per_layer,
        mlp_macs_per_layer,
        compute_macs_per_cycle,
        activation_bytes_per_position,
        link_bytes_per_cycle,
    ) <= 0:
        raise ValueError("cost parameters must be positive")
    if synchronization_cycles < 0:
        raise ValueError("synchronization_cycles must be non-negative")
    if activation_multicast_reuse <= 0 or router_cycles_per_position < 0:
        raise ValueError("reuse must be positive and router cycles non-negative")
    attention_work = max(depth_by_position) * len(depth_by_position) * attention_macs_per_layer
    mlp_work = sum(depth_by_position) * mlp_macs_per_layer
    compute_cycles = (attention_work + mlp_work) / compute_macs_per_cycle
    link_cycles = (
        len(depth_by_position)
        * activation_bytes_per_position
        / activation_multicast_reuse
        / link_bytes_per_cycle
    )
    router_cycles = len(depth_by_position) * router_cycles_per_position
    total_cycles = (
        compute_cycles + link_cycles + synchronization_cycles + router_cycles
    )
    return ChipletCost(
        compute_cycles=float(compute_cycles),
        link_cycles=float(link_cycles),
        synchronization_cycles=float(synchronization_cycles),
        total_cycles=float(total_cycles),
        router_cycles=float(router_cycles),
    )


def estimate_chiplet_width_aware_mlp_cost(
    width_by_layer: Sequence[float],
    *,
    block_size: int,
    attention_macs_per_layer: int,
    mlp_macs_per_layer: int,
    compute_macs_per_cycle: int,
    activation_bytes_per_position: int,
    link_bytes_per_cycle: int,
    synchronization_cycles: int,
    activation_multicast_reuse: float = 1.0,
    router_cycles_per_position: float = 0.0,
) -> ChipletCost:
    """Width-aware dense-attention model with explicit chiplet overheads."""

    widths = tuple(float(width) for width in width_by_layer)
    if not widths or any(not 0.0 < width <= 1.0 for width in widths):
        raise ValueError("width_by_layer values must be within (0, 1]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if min(
        attention_macs_per_layer,
        mlp_macs_per_layer,
        compute_macs_per_cycle,
        activation_bytes_per_position,
        link_bytes_per_cycle,
    ) <= 0:
        raise ValueError("cost parameters must be positive")
    if synchronization_cycles < 0:
        raise ValueError("synchronization_cycles must be non-negative")
    if activation_multicast_reuse <= 0 or router_cycles_per_position < 0:
        raise ValueError("reuse must be positive and router cycles non-negative")
    attention_work = len(widths) * block_size * attention_macs_per_layer
    mlp_work = block_size * sum(widths) * mlp_macs_per_layer
    compute_cycles = (attention_work + mlp_work) / compute_macs_per_cycle
    link_cycles = block_size * activation_bytes_per_position / (
        activation_multicast_reuse * link_bytes_per_cycle
    )
    router_cycles = block_size * router_cycles_per_position
    total_cycles = compute_cycles + link_cycles + synchronization_cycles + router_cycles
    return ChipletCost(
        compute_cycles=float(compute_cycles),
        link_cycles=float(link_cycles),
        synchronization_cycles=float(synchronization_cycles),
        total_cycles=float(total_cycles),
        router_cycles=float(router_cycles),
    )


def estimate_chiplet_cost(
    depth_by_position: Sequence[int],
    macs_per_layer: int,
    compute_macs_per_cycle: int,
    activation_bytes_per_position: int,
    link_bytes_per_cycle: int,
    synchronization_cycles: int,
    shared_lower_depth: int = 0,
    activation_multicast_reuse: float = 1.0,
    router_cycles_per_position: float = 0.0,
) -> ChipletCost:
    """Estimate a transparent upper-bound cost for split draft execution.

    The model intentionally exposes rather than hides link and synchronization
    penalties. ``shared_lower_depth`` models lower draft layers evaluated once
    for the block before positions diverge. ``activation_multicast_reuse``
    models how many positions share one activation transfer. Defaults retain
    the original independent-position upper bound.
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
    if not 0 <= shared_lower_depth <= min(depth_by_position):
        raise ValueError("shared_lower_depth must not exceed the shallowest depth")
    if activation_multicast_reuse <= 0 or router_cycles_per_position < 0:
        raise ValueError("reuse must be positive and router cycles non-negative")
    effective_layers = shared_lower_depth + sum(
        max(depth - shared_lower_depth, 0) for depth in depth_by_position
    )
    compute_cycles = effective_layers * macs_per_layer / compute_macs_per_cycle
    total_link_bytes = (
        len(depth_by_position)
        * activation_bytes_per_position
        / activation_multicast_reuse
    )
    link_cycles = total_link_bytes / link_bytes_per_cycle
    router_cycles = len(depth_by_position) * router_cycles_per_position
    total_cycles = compute_cycles + link_cycles + synchronization_cycles + router_cycles
    return ChipletCost(
        compute_cycles=float(compute_cycles),
        link_cycles=float(link_cycles),
        synchronization_cycles=float(synchronization_cycles),
        total_cycles=float(total_cycles),
        router_cycles=float(router_cycles),
    )

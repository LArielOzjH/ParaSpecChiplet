"""Analytical cost model for batch-grouped selective MLP execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GroupedMLPCost:
    batch_size: int
    block_size: int
    active_rows_by_layer: tuple[int, ...]
    schedule_groups_by_layer: tuple[int, ...]
    dense_compute_cycles: float
    dense_total_cycles: float
    grouped_compute_cycles: float
    grouped_overhead_cycles: float
    grouped_total_cycles: float
    separate_total_cycles: float


def estimate_grouped_mlp_batch_cost(
    depth_schedules: Sequence[Sequence[int]],
    *,
    draft_layers: int,
    mlp_macs_per_layer: int,
    compute_macs_per_cycle: int,
    launch_cycles: float = 0.0,
    scatter_cycles_per_row: float = 0.0,
) -> GroupedMLPCost:
    """Estimate dense, grouped, and separately-launched batch MLP costs.

    Each schedule describes one request. Grouped execution makes one active-row
    call per draft layer, while separate execution makes one call per distinct
    schedule that still has active rows in that layer. Both models pay the same
    active-row compute work; the difference is launch overhead. Scatter cost is
    charged once per active row for the grouped path.
    """

    schedules = tuple(tuple(int(depth) for depth in schedule) for schedule in depth_schedules)
    if not schedules:
        raise ValueError("depth_schedules must not be empty")
    block_size = len(schedules[0])
    if block_size == 0 or any(len(schedule) != block_size for schedule in schedules):
        raise ValueError("all schedules must have the same block size")
    if draft_layers <= 0 or any(
        depth < 1 or depth > draft_layers for schedule in schedules for depth in schedule
    ):
        raise ValueError("schedule depths must be within draft layer range")
    if mlp_macs_per_layer <= 0 or compute_macs_per_cycle <= 0:
        raise ValueError("compute parameters must be positive")
    if launch_cycles < 0 or scatter_cycles_per_row < 0:
        raise ValueError("overhead parameters must be non-negative")

    active_rows: list[int] = []
    schedule_groups: list[int] = []
    for layer_index in range(draft_layers):
        active_masks = {
            tuple(depth > layer_index for depth in schedule)
            for schedule in schedules
            if any(depth > layer_index for depth in schedule)
        }
        active_rows.append(
            sum(depth > layer_index for schedule in schedules for depth in schedule)
        )
        schedule_groups.append(len(active_masks))

    dense_compute = (
        len(schedules) * block_size * draft_layers * mlp_macs_per_layer / compute_macs_per_cycle
    )
    grouped_compute = sum(active_rows) * mlp_macs_per_layer / compute_macs_per_cycle
    active_layers = sum(rows > 0 for rows in active_rows)
    dense_total = dense_compute + draft_layers * launch_cycles
    grouped_overhead = (
        active_layers * launch_cycles + sum(active_rows) * scatter_cycles_per_row
    )
    grouped_total = grouped_compute + grouped_overhead
    separate_total = grouped_compute + sum(schedule_groups) * launch_cycles + sum(
        active_rows
    ) * scatter_cycles_per_row
    return GroupedMLPCost(
        batch_size=len(schedules),
        block_size=block_size,
        active_rows_by_layer=tuple(active_rows),
        schedule_groups_by_layer=tuple(schedule_groups),
        dense_compute_cycles=float(dense_compute),
        dense_total_cycles=float(dense_total),
        grouped_compute_cycles=float(grouped_compute),
        grouped_overhead_cycles=float(grouped_overhead),
        grouped_total_cycles=float(grouped_total),
        separate_total_cycles=float(separate_total),
    )

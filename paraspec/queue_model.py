"""Trace-driven schedule queueing summaries for grouped DFlash execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .grouped_cost import estimate_grouped_mlp_batch_cost


@dataclass(frozen=True)
class ScheduleQueueSummary:
    policy: str
    requests: int
    batches: int
    mean_batch_fill: float
    mean_schedule_groups: float
    active_row_fraction: float
    total_dense_cycles: float
    total_grouped_cycles: float
    total_separate_cycles: float


def _form_batches(labels: tuple[str, ...], capacity: int, policy: str) -> list[tuple[str, ...]]:
    if capacity <= 0:
        raise ValueError("batch capacity must be positive")
    if policy == "arrival":
        return [labels[start : start + capacity] for start in range(0, len(labels), capacity)]
    if policy == "coalesced":
        batches: list[tuple[str, ...]] = []
        remaining = list(labels)
        while remaining:
            label = remaining[0]
            same = [item for item in remaining if item == label]
            batches.extend(
                tuple(same[start : start + capacity])
                for start in range(0, len(same), capacity)
            )
            remaining = [item for item in remaining if item != label]
        return batches
    raise ValueError("policy must be 'arrival' or 'coalesced'")


def analyze_schedule_queue(
    labels: Sequence[str],
    schedules: dict[str, Sequence[int]],
    *,
    batch_capacity: int,
    draft_layers: int,
    mlp_macs_per_layer: int,
    compute_macs_per_cycle: int,
    launch_cycles: float = 0.0,
    scatter_cycles_per_row: float = 0.0,
    policy: str = "arrival",
) -> ScheduleQueueSummary:
    """Summarize queue cost for arrival-order or schedule-coalesced batches."""

    request_labels = tuple(str(label) for label in labels)
    if not request_labels:
        raise ValueError("labels must not be empty")
    if any(label not in schedules for label in request_labels):
        raise ValueError("every label must have a schedule")
    batches = _form_batches(request_labels, batch_capacity, policy)
    results = [
        estimate_grouped_mlp_batch_cost(
            tuple(schedules[label] for label in batch),
            draft_layers=draft_layers,
            mlp_macs_per_layer=mlp_macs_per_layer,
            compute_macs_per_cycle=compute_macs_per_cycle,
            launch_cycles=launch_cycles,
            scatter_cycles_per_row=scatter_cycles_per_row,
        )
        for batch in batches
    ]
    total_rows = sum(result.batch_size * result.block_size * draft_layers for result in results)
    active_rows = sum(sum(result.active_rows_by_layer) for result in results)
    return ScheduleQueueSummary(
        policy=policy,
        requests=len(request_labels),
        batches=len(batches),
        mean_batch_fill=len(request_labels) / len(batches),
        mean_schedule_groups=sum(
            sum(result.schedule_groups_by_layer) / draft_layers for result in results
        )
        / len(results),
        active_row_fraction=active_rows / total_rows,
        total_dense_cycles=sum(result.dense_total_cycles for result in results),
        total_grouped_cycles=sum(result.grouped_total_cycles for result in results),
        total_separate_cycles=sum(result.separate_total_cycles for result in results),
    )

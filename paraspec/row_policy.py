"""Hardware-side dense fallback policy from measured row-latency data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RowPolicyResult:
    nominal_active_rows_by_layer: tuple[int, ...]
    effective_active_rows_by_layer: tuple[int, ...]
    mode_by_layer: tuple[str, ...]
    saved_row_fraction: float


def evaluate_schedule_with_row_policy(
    depth_by_position: Sequence[int],
    calibration: Sequence[dict[str, float]],
    *,
    draft_layers: int,
) -> RowPolicyResult:
    """Apply an exact measured dense/grouped choice to every draft layer.

    Missing row counts conservatively use dense execution. The comparison is
    against uniform full-depth execution, so the result reports realizable
    rather than nominal row savings.
    """

    schedule = tuple(int(depth) for depth in depth_by_position)
    if not schedule or draft_layers <= 0:
        raise ValueError("schedule and draft_layers must be positive")
    if any(depth < 1 or depth > draft_layers for depth in schedule):
        raise ValueError("schedule depths must be within draft layer range")
    measured = {
        int(record["active_per_request"]): record for record in calibration
    }
    nominal: list[int] = []
    effective: list[int] = []
    modes: list[str] = []
    for layer_index in range(draft_layers):
        active = sum(depth > layer_index for depth in schedule)
        nominal.append(active)
        record = measured.get(active)
        grouped_is_better = bool(
            record is not None and float(record["grouped_ms"]) < float(record["dense_ms"])
        )
        modes.append("grouped" if grouped_is_better else "dense")
        effective.append(active if grouped_is_better else len(schedule))
    uniform_rows = len(schedule) * draft_layers
    saved = (uniform_rows - sum(effective)) / uniform_rows
    return RowPolicyResult(tuple(nominal), tuple(effective), tuple(modes), saved)

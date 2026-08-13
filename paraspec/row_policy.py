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


@dataclass(frozen=True)
class ScheduleLatencyEstimate:
    layer_latency_ms: tuple[float, ...]
    uniform_latency_ms: float
    speedup_fraction: float


def evaluate_schedule_with_row_policy(
    depth_by_position: Sequence[int],
    calibration: Sequence[dict[str, float]],
    *,
    draft_layers: int,
    batch_size: int | None = None,
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
    selected_calibration = _select_calibration(calibration, batch_size)
    measured = {
        int(record["active_per_request"]): record for record in selected_calibration
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


def _select_calibration(
    calibration: Sequence[dict[str, float]], batch_size: int | None
) -> tuple[dict[str, float], ...]:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selected = tuple(calibration)
    if batch_size is not None:
        selected = tuple(
            record
            for record in calibration
            if int(record.get("batch_size", -1)) == batch_size
        )
        if not selected:
            raise ValueError("calibration has no matching batch size")
    return selected


def estimate_schedule_mlp_latency(
    depth_by_position: Sequence[int],
    calibration: Sequence[dict[str, float]],
    *,
    draft_layers: int,
    batch_size: int | None = None,
) -> ScheduleLatencyEstimate:
    """Estimate MLP time with a measured per-layer dense fallback decision."""

    schedule = tuple(int(depth) for depth in depth_by_position)
    if not schedule or draft_layers <= 0:
        raise ValueError("schedule and draft_layers must be positive")
    if any(depth < 1 or depth > draft_layers for depth in schedule):
        raise ValueError("schedule depths must be within draft layer range")
    selected = _select_calibration(calibration, batch_size)
    by_rows = {int(record["active_per_request"]): record for record in selected}
    dense_record = by_rows.get(len(schedule))
    if dense_record is None:
        raise ValueError("calibration must contain a full-row dense measurement")
    layer_latency: list[float] = []
    for layer_index in range(draft_layers):
        active = sum(depth > layer_index for depth in schedule)
        record = by_rows.get(active)
        if record is not None and float(record["grouped_ms"]) < float(record["dense_ms"]):
            layer_latency.append(float(record["grouped_ms"]))
        else:
            layer_latency.append(float(dense_record["dense_ms"]))
    uniform_latency = float(dense_record["dense_ms"]) * draft_layers
    total = sum(layer_latency)
    return ScheduleLatencyEstimate(
        layer_latency_ms=tuple(layer_latency),
        uniform_latency_ms=uniform_latency,
        speedup_fraction=(uniform_latency - total) / uniform_latency,
    )

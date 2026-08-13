"""Measured-latency composition for fixed-shape fidelity schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CalibratedScheduleLatency:
    mlp_latency: float
    dense_attention_latency: float
    total_latency: float


def estimate_schedule_latency(
    *,
    width_by_layer: Sequence[float],
    active_rows_by_layer: Sequence[int],
    latency_table: Mapping[tuple[int, float], float],
    dense_attention_latency: float,
) -> CalibratedScheduleLatency:
    """Compose measured single-layer MLP latencies for a schedule.

    The table must come from one fixed serving regime (for example batch 64).
    This deliberately does not extrapolate across batch sizes. It is a
    transparent calibrated bound, not an end-to-end timing model.
    """

    widths = tuple(float(width) for width in width_by_layer)
    rows = tuple(int(value) for value in active_rows_by_layer)
    if len(widths) != len(rows):
        raise ValueError("width_by_layer and active_rows_by_layer must have the same length")
    if not widths:
        raise ValueError("schedule must not be empty")
    if dense_attention_latency < 0:
        raise ValueError("dense_attention_latency must be non-negative")
    values: list[float] = []
    for row_count, width in zip(rows, widths):
        if row_count <= 0 or not 0.0 < width <= 1.0:
            raise ValueError("active rows must be positive and widths within (0, 1]")
        key = (row_count, width)
        if key not in latency_table:
            raise ValueError(f"latency table has no entry for {key}")
        latency = float(latency_table[key])
        if latency < 0:
            raise ValueError("latency table values must be non-negative")
        values.append(latency)
    mlp_latency = sum(values)
    return CalibratedScheduleLatency(
        mlp_latency=mlp_latency,
        dense_attention_latency=float(dense_attention_latency),
        total_latency=mlp_latency + float(dense_attention_latency),
    )

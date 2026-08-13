"""Validation helpers for offline selective-depth DFlash probes."""

from __future__ import annotations

from typing import Sequence


def validate_depth_schedule(
    depth_by_position: Sequence[int], *, draft_layers: int, protected_prefix: int
) -> tuple[int, ...]:
    """Validate a per-position number of executed draft layers."""

    schedule = tuple(int(depth) for depth in depth_by_position)
    if not schedule:
        raise ValueError("depth schedule must not be empty")
    if draft_layers <= 0:
        raise ValueError("draft_layers must be positive")
    if not 0 <= protected_prefix <= len(schedule):
        raise ValueError("protected_prefix must be within the schedule")
    if any(depth < 1 or depth > draft_layers for depth in schedule):
        raise ValueError("depth values must be within draft layer range")
    if any(depth != draft_layers for depth in schedule[:protected_prefix]):
        raise ValueError("protected prefix must use full draft depth")
    return schedule


def skipped_positions(depth_by_position: Sequence[int], *, layer_index: int) -> tuple[bool, ...]:
    """Return positions whose update is skipped at zero-based layer index."""

    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    return tuple(depth <= layer_index for depth in depth_by_position)

"""Validation helpers for offline selective-depth DFlash probes."""

from __future__ import annotations

from typing import Sequence

import torch


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


def zero_skipped_updates(output: torch.Tensor, *, skipped: Sequence[bool]) -> torch.Tensor:
    """Zero an MLP update for skipped positions while retaining active updates."""

    if output.ndim != 3 or len(skipped) != output.shape[1]:
        raise ValueError("output must be [batch, positions, hidden] and match skipped mask")
    result = output.clone()
    mask = torch.tensor(tuple(skipped), dtype=torch.bool, device=output.device)
    result[:, mask] = 0
    return result


def selective_mlp_forward(
    mlp: object,
    hidden_states: torch.Tensor,
    *,
    skipped: Sequence[bool],
    anchors: int,
    block_size: int,
) -> torch.Tensor:
    """Evaluate a position-wise MLP only for non-skipped block rows."""

    if hidden_states.ndim != 3 or hidden_states.shape[1] != anchors * block_size:
        raise ValueError("hidden_states must be [batch, anchors * block_size, hidden]")
    if len(skipped) != block_size:
        raise ValueError("skipped mask must match block_size")
    active_one_block = ~torch.tensor(tuple(skipped), dtype=torch.bool, device=hidden_states.device)
    active = active_one_block.repeat(anchors)
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    active_rows = flat[active]
    # Calling nn.Module(...) would re-enter the hook that owns this helper.
    # Direct ``forward`` avoids that recursion while keeping lightweight test
    # doubles callable in the usual way.
    forward = getattr(mlp, "forward", None)
    active_output = forward(active_rows) if callable(forward) else mlp(active_rows)
    if active_output.shape != active_rows.shape:
        raise ValueError("MLP output shape must match its input shape")
    result = torch.zeros_like(flat)
    result[active] = active_output
    return result.view_as(hidden_states)

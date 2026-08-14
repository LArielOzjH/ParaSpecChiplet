"""Selective MLP execution helpers for the official DFlash model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import torch

from .selective_proxy import skipped_positions, validate_depth_schedule


def install_mlp_gates(
    layers: Sequence[object],
    depth_by_position: Sequence[int],
    *,
    draft_layers: int,
    dense_fallback_fraction: float | None = None,
) -> Callable[[], None]:
    """Replace each layer MLP with a position-row-selective forward path.

    Attention remains untouched. For layer ``l``, rows whose scheduled depth
    is at most ``l`` receive a zero MLP update; only active rows call the
    original MLP. If ``dense_fallback_fraction`` is provided and the active
    row fraction falls below it, the original MLP is evaluated for every row
    instead. This models an occupancy gate that avoids an unprofitable
    gather/scatter launch. The returned callback restores every original
    forward.
    """

    schedule = validate_depth_schedule(
        depth_by_position,
        draft_layers=draft_layers,
        protected_prefix=0,
    )
    if dense_fallback_fraction is not None and not 0.0 <= dense_fallback_fraction <= 1.0:
        raise ValueError("dense_fallback_fraction must be within [0, 1]")
    originals: list[tuple[object, Callable[..., torch.Tensor]]] = []
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp")
        original = getattr(mlp, "forward")
        skipped = skipped_positions(schedule, layer_index=layer_index)
        active_fraction = sum(not value for value in skipped) / len(skipped)
        use_dense_fallback = (
            dense_fallback_fraction is not None
            and active_fraction < dense_fallback_fraction
        )

        def gated_forward(
            hidden_states: torch.Tensor,
            *args: object,
            _original: Callable[..., torch.Tensor] = original,
            _skipped: tuple[bool, ...] = skipped,
            _use_dense_fallback: bool = use_dense_fallback,
            **kwargs: object,
        ) -> torch.Tensor:
            if args or kwargs:
                raise ValueError("selective MLP gate does not support extra MLP arguments")
            if hidden_states.ndim != 3 or hidden_states.shape[1] != len(_skipped):
                raise ValueError("official DFlash MLP input must be [batch, block, hidden]")
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            if _use_dense_fallback:
                dense_output = _original(flat)
                if dense_output.shape != flat.shape:
                    raise ValueError("MLP output shape must match its input shape")
                return dense_output.view_as(hidden_states)
            active = ~torch.tensor(_skipped, dtype=torch.bool, device=hidden_states.device)
            active_mask = active.repeat(hidden_states.shape[0])
            active_rows = flat[active_mask]
            result = torch.zeros_like(flat)
            if active_rows.numel():
                active_output = _original(active_rows)
                if active_output.shape != active_rows.shape:
                    raise ValueError("MLP output shape must match its input shape")
                result[active_mask] = active_output
            return result.view_as(hidden_states)

        originals.append((mlp, original))
        mlp.forward = gated_forward

    def restore() -> None:
        for mlp, original in originals:
            mlp.forward = original

    return restore

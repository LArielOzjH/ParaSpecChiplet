"""Reduced-width gated MLP helpers for acceptance and CUDA probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


def reduced_gated_mlp(hidden_states: torch.Tensor, mlp: object, intermediate_width: int) -> torch.Tensor:
    """Evaluate a Qwen-style gated MLP using a truncated intermediate width."""

    gate_proj = getattr(mlp, "gate_proj")
    up_proj = getattr(mlp, "up_proj")
    down_proj = getattr(mlp, "down_proj")
    available_width = int(gate_proj.weight.shape[0])
    if not 1 <= intermediate_width <= available_width:
        raise ValueError("intermediate_width must be within the gate projection width")
    if int(up_proj.weight.shape[0]) != available_width:
        raise ValueError("gate and up projections must have the same width")
    if int(down_proj.weight.shape[1]) != available_width:
        raise ValueError("down projection width must match gate projection width")
    gate = F.linear(hidden_states, gate_proj.weight[:intermediate_width])
    up = F.linear(hidden_states, up_proj.weight[:intermediate_width])
    return F.linear(F.silu(gate) * up, down_proj.weight[:, :intermediate_width])


def install_mlp_widths(
    layers: Sequence[object],
    width_fractions: Mapping[int, float],
    *,
    draft_layers: int,
) -> callable:
    """Replace selected MLP forwards with truncated-intermediate-width paths."""

    if not width_fractions:
        raise ValueError("width_fractions must not be empty")
    if len(layers) != draft_layers:
        raise ValueError("layers length must match draft_layers")
    indices = tuple(sorted(int(index) for index in width_fractions))
    if any(index < 0 or index >= draft_layers for index in indices):
        raise ValueError("layer index must be within draft layer range")
    fractions = {index: float(width_fractions[index]) for index in indices}
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions.values()):
        raise ValueError("width fractions must be within (0, 1]")
    originals: list[tuple[object, object]] = []
    for index in indices:
        mlp = getattr(layers[index], "mlp")
        original = getattr(mlp, "forward")
        available_width = int(mlp.gate_proj.weight.shape[0])
        width = max(1, int(available_width * fractions[index]))

        def reduced_forward(
            hidden_states: torch.Tensor,
            *args: object,
            _mlp: object = mlp,
            _width: int = width,
            **kwargs: object,
        ) -> torch.Tensor:
            if args or kwargs:
                raise ValueError("reduced-width MLP does not support extra arguments")
            return reduced_gated_mlp(hidden_states, _mlp, _width)

        originals.append((mlp, original))
        mlp.forward = reduced_forward

    def restore() -> None:
        for mlp, original in originals:
            mlp.forward = original

    return restore

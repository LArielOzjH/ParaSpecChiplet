"""Small, framework-independent helpers for draft-block ablation probes."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def validate_layer_indices(
    layer_indices: Sequence[int], *, draft_layers: int
) -> tuple[int, ...]:
    """Validate and canonicalize zero-based draft layer indices."""

    if draft_layers <= 0:
        raise ValueError("draft_layers must be positive")
    indices = tuple(sorted({int(index) for index in layer_indices}))
    if any(index < 0 or index >= draft_layers for index in indices):
        raise ValueError("layer index must be within draft layer range")
    return indices


def bypass_layer_output(layer_input: torch.Tensor, layer_output: object) -> object:
    """Replace a layer's transformed hidden state with its input.

    DFlash implementations may return either a tensor or a tuple whose first
    element is the transformed hidden state. Auxiliary outputs are preserved.
    """

    if isinstance(layer_output, torch.Tensor):
        return layer_input
    if isinstance(layer_output, tuple):
        if not layer_output or not isinstance(layer_output[0], torch.Tensor):
            raise TypeError("layer output tuple must start with a tensor")
        return (layer_input, *layer_output[1:])
    if isinstance(layer_output, list):
        if not layer_output or not isinstance(layer_output[0], torch.Tensor):
            raise TypeError("layer output list must start with a tensor")
        return [layer_input, *layer_output[1:]]
    raise TypeError("layer output must be a tensor, tuple, or list")


def install_layer_bypasses(
    layers: Sequence[object],
    layer_indices: Sequence[int],
    *,
    draft_layers: int,
) -> callable:
    """Install hooks that bypass selected draft layers and return a restore callback."""

    indices = validate_layer_indices(layer_indices, draft_layers=draft_layers)
    if len(layers) != draft_layers:
        raise ValueError("layers length must match draft_layers")
    handles = []
    for index in indices:
        layer = layers[index]

        def bypass_hook(_module: object, inputs: tuple[object, ...], output: object) -> object:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError("draft layer hook must receive hidden states as its first input")
            return bypass_layer_output(inputs[0], output)

        register = getattr(layer, "register_forward_hook", None)
        if not callable(register):
            raise TypeError("draft layers must support forward hooks")
        handles.append(register(bypass_hook))

    def restore() -> None:
        for handle in handles:
            handle.remove()

    return restore

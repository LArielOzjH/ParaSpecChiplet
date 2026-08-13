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

        def bypass_hook(
            _module: object,
            inputs: tuple[object, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> object:
            layer_input = kwargs.get("hidden_states")
            if layer_input is None and inputs:
                layer_input = inputs[0]
            if not isinstance(layer_input, torch.Tensor):
                raise TypeError("draft layer hook must receive hidden states as its first input")
            return bypass_layer_output(layer_input, output)

        register = getattr(layer, "register_forward_hook", None)
        if not callable(register):
            raise TypeError("draft layers must support forward hooks")
        handles.append(register(bypass_hook, with_kwargs=True))

    def restore() -> None:
        for handle in handles:
            handle.remove()

    return restore


def _zero_module_output(module_output: object) -> object:
    if isinstance(module_output, torch.Tensor):
        return torch.zeros_like(module_output)
    if isinstance(module_output, tuple):
        if not module_output or not isinstance(module_output[0], torch.Tensor):
            raise TypeError("module output tuple must start with a tensor")
        return (torch.zeros_like(module_output[0]), *module_output[1:])
    if isinstance(module_output, list):
        if not module_output or not isinstance(module_output[0], torch.Tensor):
            raise TypeError("module output list must start with a tensor")
        return [torch.zeros_like(module_output[0]), *module_output[1:]]
    raise TypeError("module output must be a tensor, tuple, or list")


def install_mlp_bypasses(
    layers: Sequence[object],
    layer_indices: Sequence[int],
    *,
    draft_layers: int,
) -> callable:
    """Install hooks that replace selected MLP updates with zero tensors."""

    indices = validate_layer_indices(layer_indices, draft_layers=draft_layers)
    if len(layers) != draft_layers:
        raise ValueError("layers length must match draft_layers")
    handles = []
    for index in indices:
        mlp = getattr(layers[index], "mlp", None)
        register = getattr(mlp, "register_forward_hook", None)
        if not callable(register):
            raise TypeError("draft layers must expose an MLP with forward hooks")

        def bypass_hook(
            _module: object,
            _inputs: tuple[object, ...],
            _kwargs: dict[str, object],
            output: object,
        ) -> object:
            return _zero_module_output(output)

        handles.append(register(bypass_hook, with_kwargs=True))

    def restore() -> None:
        for handle in handles:
            handle.remove()

    return restore

from types import SimpleNamespace

import pytest
import torch

from paraspec.partial_mlp import install_mlp_widths, reduced_gated_mlp


def _layers():
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 6, bias=False)
            self.up_proj = torch.nn.Linear(4, 6, bias=False)
            self.down_proj = torch.nn.Linear(6, 4, bias=False)

        def forward(self, hidden_states):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    return torch.nn.ModuleList([Layer()])


def test_reduced_gated_mlp_preserves_shape_at_partial_width():
    mlp = SimpleNamespace(
        gate_proj=SimpleNamespace(weight=torch.randn(6, 4)),
        up_proj=SimpleNamespace(weight=torch.randn(6, 4)),
        down_proj=SimpleNamespace(weight=torch.randn(4, 6)),
    )
    assert reduced_gated_mlp(torch.randn(2, 3, 4), mlp, 3).shape == (2, 3, 4)


def test_install_mlp_widths_replaces_selected_forward_and_restores():
    layers = _layers()
    values = torch.randn(2, 3, 4)
    full = layers[0].mlp(values)
    restore = install_mlp_widths(layers, {0: 0.5}, draft_layers=1)
    try:
        reduced = layers[0].mlp(values)
        assert reduced.shape == full.shape
    finally:
        restore()
    assert torch.allclose(layers[0].mlp(values), full)


def test_install_mlp_widths_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="within"):
        install_mlp_widths(_layers(), {0: 1.5}, draft_layers=1)

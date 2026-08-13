import pytest
import torch

from paraspec.block_ablation import (
    bypass_layer_output,
    install_mlp_bypasses,
    install_layer_bypasses,
    validate_layer_indices,
)


def test_validate_layer_indices_returns_sorted_unique_zero_based_indices():
    assert validate_layer_indices([2, 0, 2], draft_layers=3) == (0, 2)


def test_validate_layer_indices_rejects_out_of_range_layer():
    with pytest.raises(ValueError, match="layer index"):
        validate_layer_indices([3], draft_layers=3)


def test_bypass_layer_output_preserves_input_for_tensor_output():
    layer_input = torch.tensor([[1.0, 2.0]])
    layer_output = torch.tensor([[9.0, 9.0]])
    result = bypass_layer_output(layer_input, layer_output)
    assert torch.equal(result, layer_input)


def test_bypass_layer_output_replaces_first_tensor_in_tuple():
    layer_input = torch.tensor([[1.0, 2.0]])
    layer_output = (torch.tensor([[9.0, 9.0]]), "aux")
    result = bypass_layer_output(layer_input, layer_output)
    assert torch.equal(result[0], layer_input)
    assert result[1] == "aux"


def test_install_layer_bypasses_can_restore_selected_layers():
    class Layer(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 10

    layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])
    values = torch.tensor([[2.0]])
    restore = install_layer_bypasses(layers, [1], draft_layers=3)
    try:
        assert torch.equal(layers[0](values), torch.tensor([[12.0]]))
        assert torch.equal(layers[1](values), values)
    finally:
        restore()
    assert torch.equal(layers[1](values), torch.tensor([[12.0]]))


def test_install_layer_bypasses_accepts_keyword_hidden_states():
    class Layer(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 10

    layers = torch.nn.ModuleList([Layer()])
    values = torch.tensor([[2.0]])
    restore = install_layer_bypasses(layers, [0], draft_layers=1)
    try:
        assert torch.equal(layers[0](hidden_states=values), values)
    finally:
        restore()


def test_install_mlp_bypasses_zeros_selected_mlp_update_and_restores():
    class MLP(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states + 10

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    layers = torch.nn.ModuleList([Layer(), Layer()])
    values = torch.tensor([[2.0]])
    restore = install_mlp_bypasses(layers, [1], draft_layers=2)
    try:
        assert torch.equal(layers[0].mlp(values), torch.tensor([[12.0]]))
        assert torch.equal(layers[1].mlp(values), torch.zeros_like(values))
    finally:
        restore()
    assert torch.equal(layers[1].mlp(values), torch.tensor([[12.0]]))

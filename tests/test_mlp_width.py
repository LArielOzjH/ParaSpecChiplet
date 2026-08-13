from types import SimpleNamespace

import pytest
import torch

from scripts.benchmark_mlp_width_rows import reduced_gated_mlp


def test_reduced_gated_mlp_uses_requested_intermediate_width():
    hidden = 4
    intermediate = 6
    mlp = SimpleNamespace(
        gate_proj=SimpleNamespace(weight=torch.randn(intermediate, hidden)),
        up_proj=SimpleNamespace(weight=torch.randn(intermediate, hidden)),
        down_proj=SimpleNamespace(weight=torch.randn(hidden, intermediate)),
    )
    output = reduced_gated_mlp(torch.randn(3, hidden), mlp, 2)
    assert output.shape == (3, hidden)


def test_reduced_gated_mlp_rejects_width_larger_than_projection():
    mlp = SimpleNamespace(
        gate_proj=SimpleNamespace(weight=torch.randn(4, 2)),
        up_proj=SimpleNamespace(weight=torch.randn(4, 2)),
        down_proj=SimpleNamespace(weight=torch.randn(2, 4)),
    )
    with pytest.raises(ValueError, match="within"):
        reduced_gated_mlp(torch.randn(1, 2), mlp, 5)

import torch

from paraspec.official_selective import install_mlp_gates


class FakeLayer:
    def __init__(self):
        self.calls = []

        class Mlp:
            def __init__(inner):
                inner.calls = self.calls

            def forward(inner, values):
                inner.calls.append(tuple(values.shape))
                return values + 1

            def __call__(inner, values):
                return inner.forward(values)

        self.mlp = Mlp()


def test_mlp_gate_calls_only_active_position_rows_and_restores_forward():
    layers = [FakeLayer(), FakeLayer()]
    original = layers[1].mlp.forward
    restore = install_mlp_gates(layers, (2, 2, 1, 1), draft_layers=2)

    values = torch.zeros(1, 4, 3)
    result = layers[1].mlp(values)

    assert layers[1].calls == [(2, 3)]
    assert result[:, 0].tolist() == [[1.0, 1.0, 1.0]]
    assert result[:, 1].tolist() == [[1.0, 1.0, 1.0]]
    assert result[:, 2:].tolist() == [[[0.0, 0.0, 0.0]] * 2]

    restore()
    assert layers[1].mlp.forward == original


def test_mlp_gate_uses_dense_fallback_when_active_occupancy_is_low():
    layers = [FakeLayer(), FakeLayer()]
    restore = install_mlp_gates(
        layers,
        (2, 2, 1, 1),
        draft_layers=2,
        dense_fallback_fraction=0.75,
    )

    result = layers[1].mlp(torch.zeros(1, 4, 3))

    assert layers[1].calls == [(4, 3)]
    assert result.tolist() == [[[1.0, 1.0, 1.0]] * 4]
    restore()


def test_mlp_gate_keeps_grouped_execution_when_occupancy_is_high():
    layers = [FakeLayer(), FakeLayer()]
    restore = install_mlp_gates(
        layers,
        (2, 2, 1, 1),
        draft_layers=2,
        dense_fallback_fraction=0.5,
    )

    result = layers[1].mlp(torch.zeros(1, 4, 3))

    assert layers[1].calls == [(2, 3)]
    assert result[:, :2].tolist() == [[[1.0, 1.0, 1.0]] * 2]
    assert result[:, 2:].tolist() == [[[0.0, 0.0, 0.0]] * 2]
    restore()

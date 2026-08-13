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

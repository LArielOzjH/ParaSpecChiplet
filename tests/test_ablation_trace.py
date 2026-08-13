import pytest

from paraspec.ablation import LayerPositionTrace, marginal_layer_value


def test_layer_position_trace_requires_rectangular_gain_matrix():
    trace = LayerPositionTrace.from_mapping(
        {
            "block_size": 3,
            "draft_layers": 2,
            "baseline_prefix_survival": [1.0, 0.6, 0.2],
            "survival_after_layer": [
                [1.0, 0.55, 0.15],
                [1.0, 0.6, 0.2],
            ],
        }
    )

    assert trace.survival_after_layer[0][1] == 0.55


def test_marginal_layer_value_is_measured_against_previous_depth():
    trace = LayerPositionTrace.from_mapping(
        {
            "block_size": 2,
            "draft_layers": 2,
            "baseline_prefix_survival": [1.0, 0.5],
            "survival_after_layer": [[1.0, 0.4], [1.0, 0.5]],
        }
    )

    result = marginal_layer_value(trace)
    assert result[0] == pytest.approx((0.0, -0.1))
    assert result[1] == pytest.approx((0.0, 0.1))

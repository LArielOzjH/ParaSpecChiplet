import pytest

from paraspec.latency_bound import estimate_attention_preserving_bound


def test_attention_preserving_bound_keeps_dense_attention_cost():
    result = estimate_attention_preserving_bound(
        uniform_mlp_latency=10.0,
        scheduled_mlp_latency=(10.0, 10.0, 8.0),
        attention_latency_per_layer=5.0,
    )

    assert result.uniform_total_latency == pytest.approx(45.0)
    assert result.scheduled_total_latency == pytest.approx(43.0)
    assert result.speedup_fraction == pytest.approx(2 / 45)


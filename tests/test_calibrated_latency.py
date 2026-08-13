import pytest

from paraspec.calibrated_latency import estimate_schedule_latency


TABLE = {
    (16, 1.0): 100.0,
    (16, 0.5): 50.0,
    (9, 1.0): 80.0,
    (9, 0.5): 40.0,
}


def test_estimate_schedule_latency_sums_per_layer_calibrated_costs():
    result = estimate_schedule_latency(
        width_by_layer=(1.0, 0.5),
        active_rows_by_layer=(16, 9),
        latency_table=TABLE,
        dense_attention_latency=20.0,
    )
    assert result.mlp_latency == 140.0
    assert result.total_latency == 160.0


def test_estimate_schedule_latency_falls_back_when_row_count_missing():
    with pytest.raises(ValueError, match="latency table"):
        estimate_schedule_latency(
            width_by_layer=(1.0,),
            active_rows_by_layer=(8,),
            latency_table=TABLE,
            dense_attention_latency=1.0,
        )


def test_estimate_schedule_latency_rejects_mismatched_layers():
    with pytest.raises(ValueError, match="same length"):
        estimate_schedule_latency(
            width_by_layer=(1.0,),
            active_rows_by_layer=(16, 9),
            latency_table=TABLE,
            dense_attention_latency=1.0,
        )

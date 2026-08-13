from paraspec.row_policy import (
    estimate_schedule_mlp_latency,
    evaluate_schedule_with_row_policy,
)


CALIBRATION = (
    {"active_per_request": 4, "dense_ms": 10.0, "grouped_ms": 11.0},
    {"active_per_request": 2, "dense_ms": 10.0, "grouped_ms": 7.0},
)


def test_row_policy_falls_back_to_dense_for_unprofitable_layer():
    result = evaluate_schedule_with_row_policy((2, 2, 1, 1), CALIBRATION, draft_layers=2)

    assert result.nominal_active_rows_by_layer == (4, 2)
    assert result.effective_active_rows_by_layer == (4, 2)
    assert result.mode_by_layer == ("dense", "grouped")
    assert result.saved_row_fraction == 0.25


def test_row_policy_selects_calibration_for_requested_batch_size():
    calibration = (
        {"batch_size": 1, "active_per_request": 2, "dense_ms": 10.0, "grouped_ms": 12.0},
        {"batch_size": 64, "active_per_request": 2, "dense_ms": 10.0, "grouped_ms": 8.0},
    )
    small = evaluate_schedule_with_row_policy(
        (2, 2, 1, 1), calibration, draft_layers=2, batch_size=1
    )
    large = evaluate_schedule_with_row_policy(
        (2, 2, 1, 1), calibration, draft_layers=2, batch_size=64
    )

    assert small.mode_by_layer == ("dense", "dense")
    assert large.mode_by_layer == ("dense", "grouped")


def test_schedule_latency_uses_dense_fallback_per_layer():
    calibration = (
        {"batch_size": 64, "active_per_request": 4, "dense_ms": 10.0, "grouped_ms": 11.0},
        {"batch_size": 64, "active_per_request": 2, "dense_ms": 10.0, "grouped_ms": 7.0},
    )
    result = estimate_schedule_mlp_latency(
        (2, 2, 1, 1), calibration, draft_layers=2, batch_size=64
    )

    assert result.layer_latency_ms == (10.0, 7.0)
    assert result.uniform_latency_ms == 20.0
    assert result.speedup_fraction == 0.15

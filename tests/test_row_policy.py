from paraspec.row_policy import evaluate_schedule_with_row_policy


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

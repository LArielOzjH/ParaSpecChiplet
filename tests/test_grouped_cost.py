import pytest

from paraspec.grouped_cost import estimate_grouped_mlp_batch_cost


def test_grouped_cost_counts_active_rows_and_schedule_groups():
    result = estimate_grouped_mlp_batch_cost(
        ((2, 2, 2, 2), (2, 2, 1, 1)),
        draft_layers=2,
        mlp_macs_per_layer=100,
        compute_macs_per_cycle=100,
        launch_cycles=3,
        scatter_cycles_per_row=0.5,
    )

    assert result.batch_size == 2
    assert result.block_size == 4
    assert result.active_rows_by_layer == (8, 6)
    assert result.schedule_groups_by_layer == (1, 2)
    assert result.grouped_compute_cycles == pytest.approx(14.0)
    assert result.grouped_total_cycles == pytest.approx(27.0)
    assert result.separate_total_cycles == pytest.approx(30.0)


def test_grouped_cost_rejects_mismatched_schedules():
    with pytest.raises(ValueError, match="same block size"):
        estimate_grouped_mlp_batch_cost(
            ((2, 2), (2, 2, 2)),
            draft_layers=2,
            mlp_macs_per_layer=1,
            compute_macs_per_cycle=1,
        )

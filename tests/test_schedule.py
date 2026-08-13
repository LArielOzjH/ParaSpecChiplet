from paraspec.schedule import staircase_depth, schedule_efficiency


def test_staircase_keeps_prefix_full_depth_and_tapers_tail():
    assert staircase_depth(block_size=4, full_depth=8, min_depth=2) == (8, 6, 4, 2)


def test_schedule_efficiency_uses_prefix_survival_as_value():
    result = schedule_efficiency(
        survival=(1.0, 0.5, 0.25),
        depth_by_position=(4, 3, 2),
        cost_per_layer=1.0,
    )
    assert result.expected_committed_value == 1.75
    assert result.compute_cost == 9.0
    assert result.value_per_cost == 1.75 / 9.0


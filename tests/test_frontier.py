import pytest

from paraspec.ablation import LayerPositionTrace
from paraspec.chiplet_cost import ChipletCost
from paraspec.frontier import enumerate_depth_schedules, pareto_frontier


def test_enumerate_depth_schedules_is_monotone_and_prefix_protected():
    schedules = enumerate_depth_schedules(
        block_size=4,
        draft_layers=3,
        min_depth=1,
        protected_prefix=2,
    )

    assert schedules == (
        (3, 3, 1, 1),
        (3, 3, 2, 1),
        (3, 3, 2, 2),
        (3, 3, 3, 1),
        (3, 3, 3, 2),
        (3, 3, 3, 3),
    )


def test_pareto_frontier_uses_additive_layer_position_survival_and_cost():
    trace = LayerPositionTrace.from_mapping(
        {
            "block_size": 3,
            "draft_layers": 2,
            "baseline_prefix_survival": [0.8, 0.4, 0.1],
            "survival_after_layer": [
                [0.9, 0.5, 0.15],
                [1.0, 0.7, 0.3],
            ],
        }
    )

    frontier = pareto_frontier(
        trace,
        depth_schedules=((1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2)),
        cost_kwargs={
            "macs_per_layer": 100,
            "compute_macs_per_cycle": 100,
            "activation_bytes_per_position": 64,
            "link_bytes_per_cycle": 128,
            "synchronization_cycles": 1,
        },
    )

    assert [item.depth_by_position for item in frontier] == [
        (1, 1, 1),
        (2, 1, 1),
        (2, 2, 1),
        (2, 2, 2),
    ]
    assert frontier[0].predicted_value == pytest.approx(1.55)
    assert frontier[-1].cost.total_cycles == pytest.approx(8.5)

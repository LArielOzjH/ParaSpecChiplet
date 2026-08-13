import pytest

from paraspec.fidelity_frontier import schedule_frontier


def test_schedule_frontier_filters_by_acceptance_and_latency():
    points = schedule_frontier(
        {
            "uniform": (1.45, (1.0, 1.0)),
            "layer2_half": (1.44, (1.0, 0.5)),
            "layer23_half": (1.31, (1.0, 0.5, 0.5)),
            "unsafe": (1.0, (0.5, 0.5)),
        },
        mlp_latency_by_width={1.0: 100.0, 0.5: 50.0},
        attention_latency=100.0,
        min_accepted_prefix=1.4,
    )

    assert [(point.name, point.total_latency) for point in points] == [
        ("layer2_half", 250.0),
        ("uniform", 300.0),
    ]


def test_schedule_frontier_rejects_unknown_width():
    with pytest.raises(ValueError, match="latency"):
        schedule_frontier(
            {"bad": (1.4, (0.25,))},
            mlp_latency_by_width={1.0: 100.0},
            attention_latency=100.0,
            min_accepted_prefix=1.0,
        )


def test_schedule_frontier_requires_positive_safety_threshold():
    with pytest.raises(ValueError, match="min_accepted_prefix"):
        schedule_frontier(
            {"uniform": (1.4, (1.0,))},
            mlp_latency_by_width={1.0: 100.0},
            attention_latency=100.0,
            min_accepted_prefix=0.0,
        )

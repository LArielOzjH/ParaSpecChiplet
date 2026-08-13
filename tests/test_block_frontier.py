import pytest

from paraspec.block_frontier import empirical_block_frontier


def test_empirical_block_frontier_filters_unsafe_and_removes_dominated_groups():
    frontier = empirical_block_frontier(
        {
            (): 1.45,
            (2,): 1.34,
            (3,): 1.28,
            (2, 3): 1.09,
            (2, 3, 4): 0.77,
        },
        draft_layers=5,
        min_survival=1.2,
    )

    assert [(point.bypassed_layers, point.nominal_mlp_work) for point in frontier] == [
        ((2,), 4),
        ((), 5),
    ]


def test_empirical_block_frontier_rejects_invalid_groups():
    with pytest.raises(ValueError, match="layer index"):
        empirical_block_frontier({(5,): 1.0}, draft_layers=5, min_survival=1.0)


def test_empirical_block_frontier_requires_positive_threshold():
    with pytest.raises(ValueError, match="min_survival"):
        empirical_block_frontier({(): 1.0}, draft_layers=2, min_survival=0.0)

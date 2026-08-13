import pytest

from paraspec.survival_bounds import prefix_survival_bounds


def test_marginal_accuracy_gives_frechet_prefix_survival_bounds():
    bounds = prefix_survival_bounds((0.8, 0.5, 0.2))

    assert bounds.lower == pytest.approx((0.8, 0.3, 0.0))
    assert bounds.upper == pytest.approx((0.8, 0.5, 0.2))


def test_full_block_accuracy_tightens_all_prefix_lower_bounds():
    bounds = prefix_survival_bounds((0.8, 0.5, 0.2), full_block_accuracy=0.1)

    assert bounds.lower == pytest.approx((0.8, 0.3, 0.1))


def test_bounds_validate_accuracy_inputs():
    with pytest.raises(ValueError, match="accuracy"):
        prefix_survival_bounds((0.8, 1.1))


def test_full_block_accuracy_must_be_consistent_with_marginals():
    with pytest.raises(ValueError, match="cannot exceed"):
        prefix_survival_bounds((0.8, 0.5), full_block_accuracy=0.6)

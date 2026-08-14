import pytest

from paraspec.chiplet_boundary import chiplet_break_even_parallel_gain


def test_break_even_gain_is_one_when_chiplet_has_no_overhead():
    assert chiplet_break_even_parallel_gain(compute_cycles=100, overhead_cycles=0) == 1.0


def test_break_even_gain_includes_link_router_and_sync_overhead():
    assert chiplet_break_even_parallel_gain(
        compute_cycles=100,
        overhead_cycles=20,
    ) == pytest.approx(1.2)


def test_break_even_gain_rejects_invalid_cycles():
    with pytest.raises(ValueError, match="positive"):
        chiplet_break_even_parallel_gain(compute_cycles=0, overhead_cycles=1)
    with pytest.raises(ValueError, match="non-negative"):
        chiplet_break_even_parallel_gain(compute_cycles=1, overhead_cycles=-1)

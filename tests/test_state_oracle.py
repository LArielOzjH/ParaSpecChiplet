import pytest

from paraspec.state_oracle import (
    StateTrace,
    conditional_prefix_survival,
    expected_committed_value,
    transition_counts,
)


def test_state_trace_conditions_block_survival_on_previous_block_state():
    trace = StateTrace.from_sequences(
        acceptance_lengths=(3, 0, 2, 1),
        state_by_cycle=("high", "high", "low", "low"),
        block_size=3,
    )

    assert conditional_prefix_survival(trace) == {
        "high": (0.5, 0.5, 0.5),
        "low": (1.0, 0.5, 0.0),
    }
    assert expected_committed_value(trace) == {"high": 1.5, "low": 1.5}


def test_transition_counts_reports_state_to_next_acceptance_bucket():
    trace = StateTrace.from_sequences(
        acceptance_lengths=(3, 0, 2),
        state_by_cycle=("s0", "s3", "s0"),
        block_size=3,
    )

    assert transition_counts(trace) == {
        ("s0", 3): 1,
        ("s3", 0): 1,
        ("s0", 2): 1,
    }


def test_state_trace_validates_aligned_inputs():
    with pytest.raises(ValueError, match="same length"):
        StateTrace.from_sequences(
            acceptance_lengths=(1,), state_by_cycle=(), block_size=2
        )

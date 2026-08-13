import pytest

from paraspec.controller import ScheduleOption, choose_schedule
from paraspec.state_oracle import StateTrace
from paraspec.trace_oracle import Trace
from paraspec.controller import choose_schedule_by_state


def test_controller_rejects_schedule_that_harms_protected_prefix():
    history = Trace.from_acceptance_lengths([4, 3, 2, 1], block_size=4)
    options = (
        ScheduleOption("uniform", (3, 3, 3, 3), (1.0, 0.75, 0.5, 0.25), 12.0),
        ScheduleOption("unsafe-tail", (3, 2, 1, 1), (1.0, 0.55, 0.3, 0.2), 7.0),
    )

    decision = choose_schedule(
        history,
        options,
        protected_prefix=2,
        max_prefix_drop=0.1,
    )

    assert decision.name == "uniform"
    assert decision.rejected == ("unsafe-tail",)


def test_controller_selects_best_value_per_cost_among_safe_options():
    history = Trace.from_acceptance_lengths([4, 3, 2, 1], block_size=4)
    options = (
        ScheduleOption("uniform", (3, 3, 3, 3), (1.0, 0.75, 0.5, 0.25), 12.0),
        ScheduleOption("safe-tail", (3, 3, 2, 1), (1.0, 0.75, 0.5, 0.2), 9.0),
    )

    decision = choose_schedule(history, options, protected_prefix=2, max_prefix_drop=0.1)

    assert decision.name == "safe-tail"
    assert decision.score == pytest.approx((1.0 + 0.75 + 0.5 + 0.2) / 9.0)


def test_controller_uses_different_schedule_for_different_block_states():
    history = StateTrace.from_sequences(
        acceptance_lengths=(2, 4, 3, 1),
        state_by_cycle=("easy", "hard", "hard", "easy"),
        block_size=4,
    )
    uniform_easy = ScheduleOption(
        "uniform", (3, 3, 3, 3), (1.0, 0.75, 0.5, 0.25), 12.0
    )
    cheap_easy = ScheduleOption(
        "cheap", (3, 3, 2, 1), (1.0, 0.75, 0.5, 0.2), 9.0
    )
    uniform_hard = ScheduleOption(
        "uniform", (3, 3, 3, 3), (1.0, 1.0, 0.5, 0.25), 12.0
    )

    decisions = choose_schedule_by_state(
        history,
        options_by_state={
            "easy": (uniform_easy, cheap_easy),
            "hard": (uniform_hard,),
        },
        protected_prefix=2,
        max_prefix_drop=0.1,
    )

    assert decisions["easy"].name == "cheap"
    assert decisions["hard"].name == "uniform"

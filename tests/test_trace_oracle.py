from paraspec.trace_oracle import Trace, prefix_survival, conditional_hazard, value_density


def test_prefix_survival_counts_committed_prefixes():
    trace = Trace.from_acceptance_lengths([0, 1, 1, 3], block_size=3)

    assert prefix_survival(trace) == (0.75, 0.25, 0.25)


def test_conditional_hazard_identifies_first_failure_probability():
    trace = Trace.from_acceptance_lengths([0, 1, 1, 3], block_size=3)

    assert conditional_hazard(trace) == (0.25, 2 / 3, 0.0)


def test_value_density_normalizes_survival_by_extra_cost():
    survival = (1.0, 0.5, 0.25)
    costs = (2.0, 1.0, 0.5)

    assert value_density(survival, costs) == (0.5, 0.5, 0.5)


from paraspec.offline_acceptance import accepted_prefix_length


def test_accepted_prefix_length_stops_at_first_mismatch():
    assert accepted_prefix_length([1, 2, 3, 4], [1, 2, 9, 4]) == 2


def test_accepted_prefix_length_accepts_full_match_and_empty_match():
    assert accepted_prefix_length([], []) == 0
    assert accepted_prefix_length([1, 2], [1, 2]) == 2

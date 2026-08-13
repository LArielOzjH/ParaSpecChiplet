import pytest

from paraspec.selective_proxy import validate_depth_schedule, skipped_positions


def test_selective_schedule_requires_full_protected_prefix():
    schedule = validate_depth_schedule((3, 3, 2, 1), draft_layers=3, protected_prefix=2)
    assert schedule == (3, 3, 2, 1)
    assert skipped_positions(schedule, layer_index=1) == (False, False, False, True)


def test_selective_schedule_rejects_invalid_depths():
    with pytest.raises(ValueError, match="depth"):
        validate_depth_schedule((3, 2, 1), draft_layers=3, protected_prefix=2)

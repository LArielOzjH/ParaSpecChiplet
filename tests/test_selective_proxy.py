import pytest

from paraspec.selective_proxy import (
    validate_depth_schedule,
    skipped_positions,
    selective_mlp_forward,
    zero_skipped_updates,
)


def test_selective_schedule_requires_full_protected_prefix():
    schedule = validate_depth_schedule((3, 3, 2, 1), draft_layers=3, protected_prefix=2)
    assert schedule == (3, 3, 2, 1)
    assert skipped_positions(schedule, layer_index=1) == (False, False, False, True)


def test_selective_schedule_rejects_invalid_depths():
    with pytest.raises(ValueError, match="depth"):
        validate_depth_schedule((3, 2, 1), draft_layers=3, protected_prefix=2)


def test_zero_skipped_updates_preserves_active_positions():
    block = __import__("torch").arange(2 * 4 * 1, dtype=__import__("torch").float32).reshape(2, 4, 1)
    result = zero_skipped_updates(block, skipped=(False, True, False, True))
    assert result[:, 0].tolist() == block[:, 0].tolist()
    assert result[:, 1].tolist() == [[0.0], [0.0]]


def test_selective_mlp_forward_only_evaluates_active_rows():
    import torch

    calls = []

    class FakeMlp:
        def __call__(self, values):
            calls.append(tuple(values.shape))
            return values + 1

    values = torch.zeros(1, 4, 2)
    result = selective_mlp_forward(
        FakeMlp(), values, skipped=(False, True, False, True), anchors=1, block_size=4
    )

    assert calls == [(2, 2)]
    assert result[:, [0, 2]].tolist() == [[[1.0, 1.0], [1.0, 1.0]]]
    assert result[:, [1, 3]].tolist() == [[[0.0, 0.0], [0.0, 0.0]]]

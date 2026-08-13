import pytest

from paraspec.capture_config import resolve_block_size


def test_resolve_block_size_infers_checkpoint_default():
    assert resolve_block_size(None, 8) == 8


def test_resolve_block_size_accepts_matching_override():
    assert resolve_block_size(8, 8) == 8


def test_resolve_block_size_rejects_mismatched_override():
    with pytest.raises(ValueError, match="does not match"):
        resolve_block_size(16, 8)

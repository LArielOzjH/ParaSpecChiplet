"""Validation helpers for official DFlash capture configuration."""

from __future__ import annotations


def resolve_block_size(requested: int | None, checkpoint_block_size: int) -> int:
    """Use the checkpoint block size unless an identical override is given."""

    if checkpoint_block_size <= 0:
        raise ValueError("checkpoint block size must be positive")
    if requested is None:
        return checkpoint_block_size
    if requested <= 0:
        raise ValueError("requested block size must be positive")
    if requested != checkpoint_block_size:
        raise ValueError(
            f"requested block size {requested} does not match checkpoint block size "
            f"{checkpoint_block_size}"
        )
    return requested

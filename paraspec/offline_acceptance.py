"""Small helpers for explicitly labeled offline DFlash acceptance probes."""

from __future__ import annotations

from typing import Sequence


def accepted_prefix_length(draft_tokens: Sequence[int], target_tokens: Sequence[int]) -> int:
    """Count matching tokens before the first mismatch."""

    if len(draft_tokens) != len(target_tokens):
        raise ValueError("draft and target sequences must have equal length")
    accepted = 0
    for draft, target in zip(draft_tokens, target_tokens):
        if int(draft) != int(target):
            break
        accepted += 1
    return accepted

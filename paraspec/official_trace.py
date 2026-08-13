"""Convert official DFlash generation statistics into the local trace schema."""

from __future__ import annotations

from typing import Mapping, Sequence


def stats_to_verification_events(
    *,
    request_id: str,
    block_size: int,
    committed_tokens_per_cycle: Sequence[int],
    draft_layers: int | None = None,
    stage_latency_us: Mapping[str, float] | None = None,
) -> list[dict]:
    """Build events from DFlash stats, excluding the verifier fallback token.

    The official generator reports ``acceptance_length + 1``: accepted draft
    tokens followed by the one target-sampled fallback/continuation token.
    The local ``accepted_prefix`` field describes only accepted draft positions,
    so one is subtracted here.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    events = []
    for cycle, committed in enumerate(committed_tokens_per_cycle):
        committed = int(committed)
        if not 1 <= committed <= block_size:
            raise ValueError("committed token count must be in [1, block_size]")
        event = {
            "kind": "dflash_verify",
            "request_id": request_id,
            "cycle": cycle,
            "block_size": block_size,
            "accepted_prefix": committed - 1,
            "committed_tokens": committed,
        }
        if draft_layers is not None:
            event["draft_layers"] = int(draft_layers)
        if stage_latency_us is not None:
            event["stage_latency_us"] = dict(stage_latency_us)
        events.append(event)
    if not events:
        raise ValueError("at least one cycle is required")
    return events

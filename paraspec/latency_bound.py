"""Cycle bound for dense-attention, selectively executed MLP layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AttentionPreservingBound:
    uniform_total_latency: float
    scheduled_total_latency: float
    speedup_fraction: float


def estimate_attention_preserving_bound(
    *,
    uniform_mlp_latency: float,
    scheduled_mlp_latency: Sequence[float],
    attention_latency_per_layer: float,
) -> AttentionPreservingBound:
    """Combine measured MLP schedule times with dense attention cost."""

    if uniform_mlp_latency < 0 or attention_latency_per_layer < 0:
        raise ValueError("latencies must be non-negative")
    scheduled = tuple(float(value) for value in scheduled_mlp_latency)
    if not scheduled:
        raise ValueError("scheduled_mlp_latency must not be empty")
    if any(value < 0 for value in scheduled):
        raise ValueError("latencies must be non-negative")
    uniform_total = len(scheduled) * (uniform_mlp_latency + attention_latency_per_layer)
    scheduled_total = len(scheduled) * attention_latency_per_layer + sum(scheduled)
    speedup = 0.0 if uniform_total == 0 else (uniform_total - scheduled_total) / uniform_total
    return AttentionPreservingBound(uniform_total, scheduled_total, speedup)

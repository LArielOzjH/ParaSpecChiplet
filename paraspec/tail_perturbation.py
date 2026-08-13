"""Pure helpers for multi-condition tail perturbation experiments."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

import torch


PERTURBATION_MODES = ("mean", "copy_previous", "zero")


def perturb_tail(
    block: torch.Tensor, *, protected_prefix: int, mode: str
) -> torch.Tensor:
    """Return a copy whose unprotected tail follows the requested perturbation.

    ``block`` is shaped ``[batch, positions, hidden]``. The protected prefix is
    never modified; this makes the helper suitable for a forward hook that
    probes tail-to-prefix interference.
    """

    if block.ndim != 3:
        raise ValueError("block must have shape [batch, positions, hidden]")
    if not 0 <= protected_prefix <= block.shape[1]:
        raise ValueError("protected_prefix must be within the block")
    if mode not in PERTURBATION_MODES:
        raise ValueError(f"unsupported perturbation mode: {mode}")

    result = block.clone()
    if protected_prefix == block.shape[1]:
        return result
    tail = result[:, protected_prefix:]
    if mode == "mean":
        tail.copy_(tail.mean(dim=1, keepdim=True))
    elif mode == "copy_previous":
        source = result[:, protected_prefix - 1 : protected_prefix]
        tail.copy_(source)
    else:
        tail.zero_()
    return result


def summarize_perturbation_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate scalar perturbation measurements by experimental condition."""

    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record["mode"]), int(record["protected_prefix"]), int(record["after_layer"]))
        groups[key].append(record)

    summary = []
    for (mode, protected_prefix, after_layer), group in sorted(
        groups.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])
    ):
        summary.append(
            {
                "after_layer": after_layer,
                "mean_cosine": mean(float(item["cosine"]) for item in group),
                "mean_relative_l2": mean(float(item["relative_l2"]) for item in group),
                "mode": mode,
                "protected_prefix": protected_prefix,
                "samples": len(group),
            }
        )
    return summary

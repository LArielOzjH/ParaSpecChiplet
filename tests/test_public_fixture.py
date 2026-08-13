import json
from pathlib import Path

from paraspec.public_fixture import load_position_accuracy


def test_public_fixture_preserves_position_order_and_source():
    path = Path("data/qwen3_0.6b_dflash_shift_b8_val_metrics.json")
    result = load_position_accuracy(path)

    assert result.model == "Qwen3-0.6B-DFlash-shift-b8"
    assert result.source.startswith("https://huggingface.co/")
    assert len(result.accuracy_by_position) == 7
    assert result.accuracy_by_position[0] > result.accuracy_by_position[-1]


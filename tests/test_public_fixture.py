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


def test_target_vocab_microcycle_fixture_is_marked_as_non_survival_evidence():
    path = Path("data/qwen3_0.6b_microcycle_dflash_val_metrics.json")
    payload = json.loads(path.read_text())

    assert payload["draft_vocab_size"] == 151936
    assert payload["block_size"] == 8
    assert "not prefix-survival" in payload["interpretation_warning"]

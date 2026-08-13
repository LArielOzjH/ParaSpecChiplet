import torch

from paraspec.tail_perturbation import perturb_tail, summarize_perturbation_records


def test_perturb_tail_preserves_protected_prefix_for_supported_modes():
    block = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    for mode in ("mean", "copy_previous", "zero"):
        perturbed = perturb_tail(block, protected_prefix=2, mode=mode)
        assert torch.equal(perturbed[:, :2], block[:, :2])
        assert perturbed.shape == block.shape


def test_perturb_tail_mean_replaces_each_tail_with_block_tail_mean():
    block = torch.tensor([[[1.0], [2.0], [10.0], [14.0]]])
    perturbed = perturb_tail(block, protected_prefix=2, mode="mean")
    assert torch.equal(perturbed, torch.tensor([[[1.0], [2.0], [12.0], [12.0]]]))


def test_summarize_perturbation_records_groups_by_condition():
    records = [
        {"mode": "zero", "protected_prefix": 2, "after_layer": 1, "cosine": 0.9, "relative_l2": 0.2},
        {"mode": "zero", "protected_prefix": 2, "after_layer": 1, "cosine": 1.0, "relative_l2": 0.0},
        {"mode": "mean", "protected_prefix": 4, "after_layer": 2, "cosine": 0.8, "relative_l2": 0.4},
    ]
    summary = summarize_perturbation_records(records)
    assert summary == [
        {
            "after_layer": 1,
            "mean_cosine": 0.95,
            "mean_relative_l2": 0.1,
            "mode": "zero",
            "protected_prefix": 2,
            "samples": 2,
        },
        {
            "after_layer": 2,
            "mean_cosine": 0.8,
            "mean_relative_l2": 0.4,
            "mode": "mean",
            "protected_prefix": 4,
            "samples": 1,
        },
    ]

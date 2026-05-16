import sys
import types

import torch

from turboquant_vllm import build, expert_pruning, flute_build
from turboquant_vllm.weight_quant import packed_group_bytes


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return {"input_ids": [[1, 2, 3] for _ in texts]}


def test_build_cuda_version_tuple_accepts_major_only(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12")
    assert build._cuda_version_tuple() == (12, 0)


def test_flute_build_cuda_version_tuple_accepts_major_only(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12")
    assert flute_build._cuda_version_tuple() == (12, 0)


def test_prepare_calibration_data_falls_back_when_filter_removes_all_samples(monkeypatch):
    fake_datasets = types.ModuleType("datasets")

    def fake_load_dataset(*args, **kwargs):
        return iter(
            [
                {"text": "too short"},
                {"text": "still short"},
                {"text": ""},
            ]
        )

    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    tokenizer = _FakeTokenizer()
    inputs = expert_pruning._prepare_calibration_data(
        tokenizer=tokenizer,
        num_samples=3,
        max_length=8,
        dataset_name="dummy",
        device=torch.device("cpu"),
    )

    assert len(inputs) == 3
    assert len(tokenizer.calls) == 1
    texts, kwargs = tokenizer.calls[0]
    assert len(texts) == 3
    assert all(text.startswith("The quick brown fox") for text in texts)
    assert kwargs["max_length"] == 8


def test_packed_group_bytes_matches_formula_not_naive_division():
    """Regression test for bytes_per_group bug in vllm_quant.py line 263.

    The naive formula ``group_size * bits // 8`` produces incorrect results
    for 3-bit quantization when group_size is not a multiple of 8. The correct
    formula ``((group_size + 7) // 8) * 3`` handles partial bytes correctly.

    This test verifies that packed_group_bytes() uses the correct formula.
    """
    # Test cases where naive formula would fail (group_size not multiple of 8)
    test_cases = [
        # (bits, group_size, expected_bytes)
        (3, 9, 6),   # naive: 9 * 3 // 8 = 3 (wrong), correct: ((9+7)//8)*3 = 6
        (3, 10, 6),  # naive: 10 * 3 // 8 = 3 (wrong), correct: ((10+7)//8)*3 = 6
        (3, 15, 6),  # naive: 15 * 3 // 8 = 5 (wrong), correct: ((15+7)//8)*3 = 6
        (3, 17, 9),  # naive: 17 * 3 // 8 = 6 (wrong), correct: ((17+7)//8)*3 = 9
        # Test cases where both formulas happen to match (multiples of 8)
        (3, 8, 3),
        (3, 16, 6),
        (3, 128, 48),
    ]

    for bits, group_size, expected_bytes in test_cases:
        actual_bytes = packed_group_bytes(bits, group_size)
        naive_bytes = group_size * bits // 8

        # Verify the function returns the correct value
        assert actual_bytes == expected_bytes, (
            f"packed_group_bytes({bits}, {group_size}) = {actual_bytes}, "
            f"expected {expected_bytes}"
        )

        # For non-multiples of 8, verify naive formula would have been wrong
        if group_size % 8 != 0:
            assert actual_bytes != naive_bytes, (
                f"For group_size={group_size} not divisible by 8, "
                f"packed_group_bytes should differ from naive formula"
            )

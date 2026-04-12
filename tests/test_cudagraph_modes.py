import json

import pytest

from turboquant_vllm.cudagraph_modes import (
    VALID_CUDAGRAPH_MODES,
    compilation_config_json,
    normalize_cudagraph_mode,
)


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [
        ("none", "NONE"),
        (" piecewise ", "PIECEWISE"),
        ("full", "FULL"),
        ("full-decode-only", "FULL_DECODE_ONLY"),
        ("full_and_piecewise", "FULL_AND_PIECEWISE"),
    ],
)
def test_normalize_cudagraph_mode_valid_values(raw_mode, expected):
    assert normalize_cudagraph_mode(raw_mode) == expected


def test_normalize_cudagraph_mode_uses_default_for_none():
    assert normalize_cudagraph_mode(None) == "FULL_AND_PIECEWISE"


def test_normalize_cudagraph_mode_uses_explicit_default_for_empty_input():
    assert normalize_cudagraph_mode("", default="FULL") == "FULL"


def test_normalize_cudagraph_mode_rejects_unsupported_value():
    with pytest.raises(ValueError) as exc_info:
        normalize_cudagraph_mode("invalid-mode")
    message = str(exc_info.value)
    assert "Unsupported cudagraph mode" in message
    assert "invalid-mode" in message
    for valid_mode in VALID_CUDAGRAPH_MODES:
        assert valid_mode in message


def test_compilation_config_json_normalizes_mode():
    config = json.loads(compilation_config_json("full-decode-only"))
    assert config == {"cudagraph_mode": "FULL_DECODE_ONLY"}


def test_compilation_config_json_uses_explicit_default():
    config = json.loads(compilation_config_json(None, default="FULL"))
    assert config == {"cudagraph_mode": "FULL"}

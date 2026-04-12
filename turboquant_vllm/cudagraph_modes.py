"""Helpers for vLLM CUDA graph mode configuration."""

from __future__ import annotations

import json

VALID_CUDAGRAPH_MODES: tuple[str, ...] = (
    "NONE",
    "PIECEWISE",
    "FULL",
    "FULL_DECODE_ONLY",
    "FULL_AND_PIECEWISE",
)


def normalize_cudagraph_mode(mode: str | None, default: str = "FULL_AND_PIECEWISE") -> str:
    """Return a normalized vLLM cudagraph mode string."""
    normalized = (mode or default).strip().upper().replace("-", "_")
    if normalized not in VALID_CUDAGRAPH_MODES:
        valid = ", ".join(VALID_CUDAGRAPH_MODES)
        raise ValueError(f"Unsupported cudagraph mode: {mode!r}. Expected one of: {valid}.")
    return normalized


def compilation_config_json(mode: str | None, default: str = "FULL_AND_PIECEWISE") -> str:
    """Build a vLLM --compilation-config JSON for cudagraph mode."""
    return json.dumps({"cudagraph_mode": normalize_cudagraph_mode(mode, default=default)})

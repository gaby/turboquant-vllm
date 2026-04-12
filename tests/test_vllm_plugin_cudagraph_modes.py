from __future__ import annotations

from unittest import mock

import pytest

import turboquant_vllm._vllm_plugin as plugin
from turboquant_vllm.cudagraph_modes import VALID_CUDAGRAPH_MODES


@pytest.fixture(autouse=True)
def _reset_plugin_state():
    plugin._patched = False
    yield
    plugin._patched = False


@pytest.mark.parametrize("mode", VALID_CUDAGRAPH_MODES)
def test_register_weight_path_accepts_all_valid_cudagraph_modes(monkeypatch, mode):
    monkeypatch.setenv("TQ_WEIGHT_BITS", "3")
    monkeypatch.setenv("TQ_WEIGHT_GROUP_SIZE", "128")
    monkeypatch.setenv("CUDAGRAPH_MODE", mode.lower().replace("_", "-"))

    with (
        mock.patch("turboquant_vllm.vllm_quant.register") as register_quant_config,
        mock.patch("turboquant_vllm.weight_quant.patch_vllm_loader") as patch_vllm_loader,
    ):
        plugin.register()

    register_quant_config.assert_called_once_with()
    patch_vllm_loader.assert_called_once_with(bits=3, group_size=128, min_size=128)


def test_register_rejects_invalid_cudagraph_mode(monkeypatch):
    monkeypatch.setenv("TQ_WEIGHT_BITS", "3")
    monkeypatch.setenv("CUDAGRAPH_MODE", "not-a-real-mode")

    with mock.patch("turboquant_vllm.vllm_quant.register"):
        with pytest.raises(ValueError, match="Invalid CUDAGRAPH_MODE"):
            plugin.register()


def test_register_kv_path_warns_when_cudagraph_is_enabled(monkeypatch):
    monkeypatch.setenv("TQ_KV_K_BITS", "4")
    monkeypatch.setenv("CUDAGRAPH_MODE", "FULL_AND_PIECEWISE")

    with (
        mock.patch("turboquant_vllm.vllm_quant.register"),
        mock.patch("turboquant_vllm._vllm_plugin.logger.warning") as warn,
        mock.patch("turboquant_vllm.vllm_patch.patch_vllm_attention") as patch_attention,
    ):
        plugin.register()

    patch_attention.assert_called_once_with(
        k_bits=4,
        v_bits=4,
        norm_correction=True,
        rotation="wht",
        boundary_layers=5,
    )
    legacy_warning_calls = [
        call
        for call in warn.call_args_list
        if call.args and "Legacy TurboQuant KV monkey-patch with CUDAGRAPH_MODE=%s" in str(call.args[0])
    ]
    assert legacy_warning_calls
    assert legacy_warning_calls[0].args[1] == "FULL_AND_PIECEWISE"


def test_register_kv_path_no_graph_warning_for_none_mode(monkeypatch):
    monkeypatch.setenv("TQ_KV_K_BITS", "4")
    monkeypatch.setenv("CUDAGRAPH_MODE", "none")

    with (
        mock.patch("turboquant_vllm.vllm_quant.register"),
        mock.patch("turboquant_vllm._vllm_plugin.logger.warning") as warn,
        mock.patch("turboquant_vllm.vllm_patch.patch_vllm_attention"),
    ):
        plugin.register()

    warning_messages = [str(call.args[0]) for call in warn.call_args_list if call.args]
    assert not any("Legacy TurboQuant KV monkey-patch with CUDAGRAPH_MODE=" in msg for msg in warning_messages)

"""CPU unit tests for online-MoE load-completion accounting (TP/EP fix).

The online MoE quant method buffers weight_loader calls and triggers
materialize+compress when the load is complete. Completion must be measured
in CHECKPOINT-side numel: under tensor parallelism the loader receives full
tensors and shards them into 1/tp-sized params, and under expert parallelism
the loader fires for every global expert. Measuring against the partition-side
param sum fired the trigger at ~1/tp of the load and compressed partially
loaded experts (then silently dropped the remaining loads into the shared
dequant scratch pool).

These tests exercise the module-level helpers directly — the
``TurboQuantOnlineMoEMethod`` class itself is only defined when vLLM is
installed, which CPU CI does not have.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from turboquant_vllm.vllm_quant import (
    _finish_online_moe_load,
    _moe_expected_checkpoint_numel,
)


class _Layer(nn.Module):
    pass


def _layer_with(**attrs) -> _Layer:
    layer = _Layer()
    for key, value in attrs.items():
        setattr(layer, key, value)
    return layer


# ---------------------------------------------------------------------------
# _moe_expected_checkpoint_numel
# ---------------------------------------------------------------------------


def test_expected_numel_scales_by_tp():
    layer = _layer_with(tp_size=2, global_num_experts=8, local_num_experts=8)

    assert _moe_expected_checkpoint_numel(layer, None, 1000) == 2000


def test_expected_numel_scales_by_ep():
    # EP=2: loader fires for all 8 global experts, params hold 4 local ones.
    layer = _layer_with(tp_size=1, global_num_experts=8, local_num_experts=4)

    assert _moe_expected_checkpoint_numel(layer, None, 1000) == 2000


def test_expected_numel_tp1_single_gpu_is_identity():
    layer = _layer_with(tp_size=1, global_num_experts=8, local_num_experts=8)

    assert _moe_expected_checkpoint_numel(layer, None, 1000) == 1000


def test_expected_numel_falls_back_to_moe_config():
    moe_config = types.SimpleNamespace(tp_size=4, num_experts=16, num_local_experts=16)

    assert _moe_expected_checkpoint_numel(_Layer(), moe_config, 100) == 400


def test_expected_numel_unknown_factors_returns_none():
    assert _moe_expected_checkpoint_numel(_Layer(), None, 1000) is None
    # bool attributes must not masquerade as parallel sizes
    layer = _layer_with(tp_size=True, global_num_experts=8, local_num_experts=8)
    assert _moe_expected_checkpoint_numel(layer, None, 1000) is None


def test_expected_numel_non_divisible_experts_returns_none():
    layer = _layer_with(tp_size=1, global_num_experts=7, local_num_experts=3)

    assert _moe_expected_checkpoint_numel(layer, None, 1000) is None


# ---------------------------------------------------------------------------
# _finish_online_moe_load
# ---------------------------------------------------------------------------


class _FakeMethod:
    def __init__(self):
        self.compressed = 0

    def _do_compress(self, layer):
        self.compressed += 1


def _copy_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)
    return True


def _pending_state_for(layer, buffer):
    return {
        "buffer": buffer,
        "orig_loaders": {name: _copy_loader for name, _ in layer.named_parameters(recurse=False)},
        "param_shapes": {name: tuple(p.shape) for name, p in layer.named_parameters(recurse=False)},
        "param_dtypes": {name: p.dtype for name, p in layer.named_parameters(recurse=False)},
        "materialized": [False],
    }


def test_finish_replays_buffered_loads_and_compresses():
    """Threshold never fired (e.g. unknown TP factors) — the fallback must
    materialize the meta params, replay every buffered load, then compress."""
    layer = _Layer()
    layer.register_parameter(
        "w13_weight",
        nn.Parameter(torch.empty(2, 4, 8, device="meta"), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight",
        nn.Parameter(torch.empty(2, 8, 4, device="meta"), requires_grad=False),
    )
    w13_data = torch.randn(2, 4, 8)
    w2_data = torch.randn(2, 8, 4)
    buffer = [
        ("w13_weight", (layer.w13_weight, w13_data), {}),
        ("w2_weight", (layer.w2_weight, w2_data), {}),
    ]
    state = _pending_state_for(layer, buffer)
    method = _FakeMethod()

    _finish_online_moe_load(layer, method, state)

    assert method.compressed == 1
    assert state["materialized"][0] is True
    assert not layer.w13_weight.is_meta
    torch.testing.assert_close(layer.w13_weight.data, w13_data)
    torch.testing.assert_close(layer.w2_weight.data, w2_data)
    assert not buffer  # replayed loads are cleared


def test_finish_raises_when_buffer_covers_only_some_params():
    """A non-empty buffer must not bypass the completeness check: a meta
    param with zero buffered loads would be torch.empty-materialized and
    compressed from uninitialized memory."""
    layer = _Layer()
    layer.register_parameter(
        "w13_weight",
        nn.Parameter(torch.empty(2, 4, 8, device="meta"), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight",
        nn.Parameter(torch.empty(2, 8, 4, device="meta"), requires_grad=False),
    )
    buffer = [("w2_weight", (layer.w2_weight, torch.randn(2, 8, 4)), {})]
    state = _pending_state_for(layer, buffer)
    method = _FakeMethod()

    with pytest.raises(RuntimeError, match="w13_weight"):
        _finish_online_moe_load(layer, method, state)

    assert method.compressed == 0


def test_finish_raises_on_meta_weights_with_nothing_to_replay():
    """Meta params report numel() > 0, so compressing without a replay would
    bake uninitialized data — this must be a loud error instead."""
    layer = _Layer()
    layer.register_parameter(
        "w13_weight",
        nn.Parameter(torch.empty(2, 4, 8, device="meta"), requires_grad=False),
    )
    state = _pending_state_for(layer, [])

    with pytest.raises(RuntimeError, match="meta device"):
        _finish_online_moe_load(layer, _FakeMethod(), state)


def test_finish_compresses_already_materialized_weights():
    layer = _Layer()
    layer.register_parameter(
        "w13_weight",
        nn.Parameter(torch.randn(2, 4, 8), requires_grad=False),
    )
    method = _FakeMethod()

    _finish_online_moe_load(layer, method, None)

    assert method.compressed == 1


def test_finish_skips_layer_without_moe_weights():
    method = _FakeMethod()

    _finish_online_moe_load(_Layer(), method, None)

    assert method.compressed == 0

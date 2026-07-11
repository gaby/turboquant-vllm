"""CPU unit tests for per-projection bit widths on the native-packed MoE path.

Regression: checkpoints saved with ``sensitive_bits`` pack w2 (down_proj)
at a different width than w13, but ``_finalize_native_packed_moe`` decoded
both with the uniform ``method.bits`` — a 4-bit-packed w2 read as 3-bit
fails ``Compressed3D.from_packed``'s layout check (or, with matching byte
counts, decodes garbage).
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from turboquant_vllm.vllm_quant import _finalize_native_packed_moe  # noqa: E402
from turboquant_vllm.weight_quant import Compressed3D  # noqa: E402

GROUP_SIZE = 64
N_EXPERTS = 2
W13_OUT, W2_OUT = 8, 4
IN_DIM = 64


def _fake_layer(w13_c: Compressed3D, w2_c: Compressed3D) -> nn.Module:
    layer = nn.Module()
    for name, comp in (("w13_weight", w13_c), ("w2_weight", w2_c)):
        layer.register_parameter(
            f"{name}_tq_packed",
            nn.Parameter(comp.packed, requires_grad=False),
        )
        layer.register_parameter(
            f"{name}_tq_norms",
            nn.Parameter(comp.norms, requires_grad=False),
        )
    return layer


def _fake_method(bits: int, w13_bits: int, w2_bits: int):
    method = types.SimpleNamespace()
    method.bits = bits
    method.group_size = GROUP_SIZE
    method.w13_bits = w13_bits
    method.w2_bits = w2_bits
    method._unquant = types.SimpleNamespace(process_weights_after_loading=lambda layer: None)
    pool_slot = [None]
    method._get_moe_scratch_pool = lambda: pool_slot[0]
    method._set_moe_scratch_pool = lambda p: pool_slot.__setitem__(0, p)
    return method


def _finalize(w13_bits: int, w2_bits: int, method):
    torch.manual_seed(0)
    w13_data = torch.randn(N_EXPERTS, W13_OUT, IN_DIM, dtype=torch.bfloat16)
    w2_data = torch.randn(N_EXPERTS, W2_OUT, IN_DIM, dtype=torch.bfloat16)
    w13_c = Compressed3D(w13_data, bits=w13_bits, group_size=GROUP_SIZE)
    w2_c = Compressed3D(w2_data, bits=w2_bits, group_size=GROUP_SIZE)
    layer = _fake_layer(w13_c, w2_c)
    _finalize_native_packed_moe(
        layer,
        method,
        {"w13_weight": tuple(w13_data.shape), "w2_weight": tuple(w2_data.shape)},
        {"w13_weight": torch.bfloat16, "w2_weight": torch.bfloat16},
    )
    return layer, w13_data, w2_data


class TestPerProjectionBits:
    def test_mixed_bits_decode_with_their_own_width(self):
        method = _fake_method(bits=3, w13_bits=3, w2_bits=4)
        layer, w13_data, w2_data = _finalize(3, 4, method)

        assert layer._tq_w13_weight.bits == 3
        assert layer._tq_w2_weight.bits == 4

        w13_hat = layer._tq_w13_weight.decompress().float()
        w2_hat = layer._tq_w2_weight.decompress().float()
        mse13 = ((w13_hat - w13_data.float()) ** 2).mean().item()
        mse2 = ((w2_hat - w2_data.float()) ** 2).mean().item()
        assert mse13 < 0.25, f"w13 (3-bit) MSE {mse13:.4f} too high"
        assert mse2 < 0.05, f"w2 (4-bit) MSE {mse2:.4f} too high"

    def test_uniform_bits_fallback_for_methods_without_split(self):
        # Test doubles / older methods without w13_bits/w2_bits still work
        method = _fake_method(bits=3, w13_bits=3, w2_bits=3)
        del method.w13_bits, method.w2_bits
        layer, _, _ = _finalize(3, 3, method)
        assert layer._tq_w13_weight.bits == 3
        assert layer._tq_w2_weight.bits == 3

    def test_wrong_uniform_bits_is_a_loud_error(self):
        # Documents the pre-fix failure mode: decoding a 4-bit-packed w2 at
        # the uniform 3 bits trips the packed-layout validation.
        method = _fake_method(bits=3, w13_bits=3, w2_bits=3)
        with pytest.raises(ValueError):
            _finalize(3, 4, method)

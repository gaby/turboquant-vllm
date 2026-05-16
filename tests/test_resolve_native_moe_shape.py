"""CPU tests for `_resolve_native_moe_shape` (wall #9 fix).

DeepSeek-V4-Flash registers ``w13_weight`` with the un-fused out_dim
(moe_intermediate_size) while the native TQ3 checkpoint packs the
gate_up-fused w13 (2 * moe_intermediate_size). The packed tensor is
ground truth; `_resolve_native_moe_shape` recovers the true out_dim so
both DSV4 and standard FusedMoE conventions load correctly, without
silently accepting a genuinely inconsistent layout.

No GPU / no vLLM required.
"""

from __future__ import annotations

import unittest

import torch

from turboquant_vllm.weight_quant import Compressed3D
from turboquant_vllm.vllm_quant import _resolve_native_moe_shape

BITS, GS = 3, 128
E, I, H = 4, 128, 128  # n_experts, moe_intermediate, hidden


def _fused_w13(dtype=torch.float32):
    """A real Compressed3D for the gate_up-fused w13 (out_dim = 2*I)."""
    data = torch.randn(E, 2 * I, H, dtype=dtype)
    return data, Compressed3D(data, bits=BITS, group_size=GS)


class TestResolveNativeMoEShape(unittest.TestCase):
    def test_dsv4_w13_under_reported_is_corrected(self):
        """DSV4 registers (E, I, H); packed is fused (E, 2I, H)."""
        _data, comp = _fused_w13()
        resolved = _resolve_native_moe_shape(comp.packed, (E, I, H), BITS, GS)
        self.assertEqual(resolved, (E, 2 * I, H))

    def test_corrected_shape_round_trips(self):
        """The corrected shape must yield a semantically valid Compressed3D
        (decompress matches the original), not just a shape that passes
        the asserts."""
        data, comp = _fused_w13()
        ref = comp.decompress()
        resolved = _resolve_native_moe_shape(comp.packed, (E, I, H), BITS, GS)
        rebuilt = Compressed3D.from_packed(
            comp.packed, comp.norms, resolved, data.dtype, BITS, GS
        )
        out = rebuilt.decompress()
        self.assertEqual(ref.shape, out.shape)
        self.assertTrue(
            torch.allclose(ref, out),
            f"corrected-shape decompress diverged: {(ref - out).abs().max():.6g}",
        )

    def test_standard_already_fused_unchanged(self):
        """Standard FusedMoE already registers (E, 2I, H): no change, so
        no regression for non-DSV4 models."""
        _data, comp = _fused_w13()
        resolved = _resolve_native_moe_shape(comp.packed, (E, 2 * I, H), BITS, GS)
        self.assertEqual(resolved, (E, 2 * I, H))

    def test_w2_non_gated_unchanged(self):
        """w2 (down-proj) is not gated; param out_dim matches packed."""
        data = torch.randn(E, H, 2 * I, dtype=torch.float32)  # (E, out=H, in=2I)
        comp = Compressed3D(data, bits=BITS, group_size=GS)
        resolved = _resolve_native_moe_shape(comp.packed, (E, H, 2 * I), BITS, GS)
        self.assertEqual(resolved, (E, H, 2 * I))

    def test_inconsistent_layout_not_silently_accepted(self):
        """If the packed implies an out_dim that is neither out_dim nor
        2*out_dim, return the ORIGINAL shape so the downstream validator
        raises its precise error rather than masking corruption."""
        data = torch.randn(E, 4 * I, H, dtype=torch.float32)  # true out = 4I
        comp = Compressed3D(data, bits=BITS, group_size=GS)
        # param claims out_dim = I -> true (4I) is not I and not 2I
        resolved = _resolve_native_moe_shape(comp.packed, (E, I, H), BITS, GS)
        self.assertEqual(resolved, (E, I, H))  # unchanged -> downstream raises
        with self.assertRaises(ValueError):
            Compressed3D.from_packed(
                comp.packed, comp.norms, resolved, data.dtype, BITS, GS
            )

    def test_idempotent_on_correct_shape(self):
        _data, comp = _fused_w13()
        once = _resolve_native_moe_shape(comp.packed, (E, I, H), BITS, GS)
        twice = _resolve_native_moe_shape(comp.packed, once, BITS, GS)
        self.assertEqual(once, (E, 2 * I, H))
        self.assertEqual(twice, (E, 2 * I, H))


if __name__ == "__main__":
    unittest.main()

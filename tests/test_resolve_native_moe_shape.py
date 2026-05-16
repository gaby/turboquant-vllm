"""CPU tests for `_resolve_native_moe_shape` (walls #9 + #10 fix).

DeepSeek-V4-Flash registers MoE weights with shapes that mis-report
out_dim (w13 un-fused / half) AND in_dim (w2), while the native TQ3
checkpoint packs the true gate_up-fused, full-width tensors. The norms
tensor is the authoritative quant metadata
(``(n_experts * out_dim, n_groups)``); `_resolve_native_moe_shape`
recovers the true shape from it so both DSV4 and standard FusedMoE
conventions load correctly, without masking a genuinely inconsistent
packed tensor.

No GPU / no vLLM required.
"""

from __future__ import annotations

import unittest

import torch

from turboquant_vllm.weight_quant import Compressed3D
from turboquant_vllm.vllm_quant import _resolve_native_moe_shape

BITS, GS = 3, 128
E = 4
I = 256  # moe_intermediate
H = 128  # hidden


def _comp(out_dim: int, in_dim: int, dtype=torch.float32):
    data = torch.randn(E, out_dim, in_dim, dtype=dtype)
    return data, Compressed3D(data, bits=BITS, group_size=GS)


def _resolve(comp, shape):
    return _resolve_native_moe_shape(comp.packed, comp.norms, shape, BITS, GS)


class TestResolveNativeMoEShape(unittest.TestCase):
    def test_dsv4_w13_out_dim_under_reported(self):
        """w13 packed fused (E, 2I, H); param registered (E, I, H)."""
        _d, comp = _comp(2 * I, H)
        self.assertEqual(_resolve(comp, (E, I, H)), (E, 2 * I, H))

    def test_dsv4_w2_in_dim_under_reported(self):
        """w2 packed full (E, H, I); param registered (E, H, I//2).
        This is the wall #10 case the packed-only heuristic got wrong."""
        _d, comp = _comp(H, I)
        self.assertEqual(_resolve(comp, (E, H, I // 2)), (E, H, I))

    def test_standard_shapes_unchanged(self):
        """Well-formed (standard FusedMoE) shapes: norms already matches
        the registered shape -> no-op, so no regression for non-DSV4."""
        _d, w13 = _comp(2 * I, H)
        self.assertEqual(_resolve(w13, (E, 2 * I, H)), (E, 2 * I, H))
        _d2, w2 = _comp(H, I)
        self.assertEqual(_resolve(w2, (E, H, I)), (E, H, I))

    def test_corrected_shapes_round_trip(self):
        """The resolved shape must yield a semantically valid Compressed3D
        (decompress matches the original), for both w13 and w2."""
        for data_out, data_in, bad_shape in (
            (2 * I, H, (E, I, H)),  # w13: out under-reported
            (H, I, (E, H, I // 2)),  # w2: in under-reported
        ):
            data, comp = _comp(data_out, data_in)
            ref = comp.decompress()
            resolved = _resolve(comp, bad_shape)
            rebuilt = Compressed3D.from_packed(comp.packed, comp.norms, resolved, data.dtype, BITS, GS)
            out = rebuilt.decompress()
            self.assertEqual(ref.shape, out.shape)
            self.assertTrue(
                torch.allclose(ref, out),
                f"{bad_shape}: decompress diverged {(ref - out).abs().max():.6g}",
            )

    def test_inconsistent_packed_not_silently_accepted(self):
        """If packed is inconsistent with norms, return the ORIGINAL shape
        so the downstream validator raises (no silent corruption)."""
        _d, comp = _comp(2 * I, H)
        bad_packed = comp.packed[:-1]  # numel no longer matches norms
        resolved = _resolve_native_moe_shape(bad_packed, comp.norms, (E, I, H), BITS, GS)
        self.assertEqual(resolved, (E, I, H))  # unchanged -> downstream raises

    def test_idempotent(self):
        _d, comp = _comp(2 * I, H)
        once = _resolve(comp, (E, I, H))
        twice = _resolve(comp, once)
        self.assertEqual(once, (E, 2 * I, H))
        self.assertEqual(twice, (E, 2 * I, H))


if __name__ == "__main__":
    unittest.main()

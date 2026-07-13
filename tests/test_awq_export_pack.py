"""CPU tests for the AWQ export packing (`export._compute_awq_params`).

Regression: qweight/qzeros were packed in linear column order, but the
AWQ GEMM/dequant kernels (AutoAWQ, vLLM's awq_dequantize) reverse the
interleaved order [0, 2, 4, 6, 1, 3, 5, 7] — a linear pack dequantized
to a column-permuted weight.
"""

import pytest

torch = pytest.importorskip("torch")

from turboquant_vllm.export import _compute_awq_params, awq_pack_order  # noqa: E402


def test_awq_pack_order_matches_awq_reference():
    # The canonical AWQ 4-bit shuffle, as hardcoded in AutoAWQ/vLLM kernels.
    assert awq_pack_order(8) == [0, 2, 4, 6, 1, 3, 5, 7]


def _awq_unpack(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Reference unpack mirroring vLLM's awq_dequantize shuffle."""
    pack_factor = 32 // bits
    rows, packed_cols = packed.shape
    out = torch.zeros(rows, packed_cols * pack_factor, dtype=torch.int32)
    for i, src in enumerate(awq_pack_order(pack_factor)):
        out[:, src::pack_factor] = (packed >> (i * bits)) & ((1 << bits) - 1)
    return out


class TestAwqPackOrder:
    def test_qweight_roundtrip_through_awq_order(self):
        torch.manual_seed(0)
        weight = torch.randn(16, 128)
        qweight, scales, qzeros = _compute_awq_params(weight, group_size=128, bits=4)

        assert qweight.dtype == torch.int32
        assert qweight.shape == (128, 16 // 8)  # (in, out // pack_factor)
        assert scales.dtype == torch.float16
        assert qzeros.dtype == torch.int32
        assert qzeros.shape == (1, qweight.shape[1])

        w_int = _awq_unpack(qweight)  # (in, out)
        zeros_int = _awq_unpack(qzeros)  # (n_groups, out)

        # Reconstruct: w ≈ (w_int - zeros) * scales, transposed back to (out, in)
        recon = (w_int.float() - zeros_int.float()) * scales.float()
        recon = recon.t()
        err = (recon - weight).abs().max().item()
        scale_mag = scales.float().abs().max().item()
        # Round-off is bounded by ~1 quantization step (+1 for the rounded zero)
        assert err <= 2.0 * scale_mag + 1e-4, f"AWQ dequant error {err:.5f} too high"

    def test_linear_order_would_permute(self):
        """The packed layout is genuinely interleaved (guards against
        regressing to linear order, which round-trips only through itself)."""
        torch.manual_seed(1)
        weight = torch.randn(16, 128)
        qweight, _scales, _qzeros = _compute_awq_params(weight, group_size=128, bits=4)

        linear_unpack = torch.zeros_like(_awq_unpack(qweight))
        for i in range(8):
            linear_unpack[:, i::8] = (qweight >> (i * 4)) & 0xF

        assert not torch.equal(linear_unpack, _awq_unpack(qweight))

"""Tests for TurboQuant weight quantization.

These tests validate the weight compression pipeline without vLLM,
against the current public API (pack_indices/unpack_indices,
_get_quantizer, TurboQuantWrapper). CPU-runnable so CI exercises them;
CUDA is not required.

Run: pytest tests/test_weight_quant.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from turboquant_vllm.weight_quant import (  # noqa: E402
    TurboQuantWrapper,
    _get_quantizer,
    pack_indices,
    packed_group_bytes,
    select_bits,
    unpack_indices,
)


class TestQuantizeDequantize:
    """Quantizer quantize/dequantize roundtrip quality."""

    def _roundtrip_mse(self, bits: int, group_size: int = 128) -> float:
        torch.manual_seed(42)
        w = torch.randn(256, 512)
        grouped = w.reshape(-1, group_size)
        quantizer = _get_quantizer(group_size, bits, "cpu")
        indices, norms = quantizer.quantize(grouped, norm_correction=True)
        w_hat = quantizer.dequantize(indices, norms).reshape(w.shape)
        return ((w - w_hat) ** 2).mean().item()

    def test_4bit_mse(self):
        assert self._roundtrip_mse(bits=4) < 0.02

    def test_3bit_higher_mse_than_4bit(self):
        mse3 = self._roundtrip_mse(bits=3)
        mse4 = self._roundtrip_mse(bits=4)
        assert mse4 < mse3 < 0.2, f"3-bit MSE {mse3:.6f}, 4-bit MSE {mse4:.6f}"

    def test_2bit(self):
        assert self._roundtrip_mse(bits=2) < 0.35


class TestPacking:
    """Index packing and unpacking (sub-byte layouts)."""

    @pytest.mark.parametrize(
        ("bits", "expected_cols"),
        [(4, 64), (3, 48), (2, 32)],
    )
    def test_roundtrip(self, bits, expected_cols):
        torch.manual_seed(0)
        indices = torch.randint(0, 2**bits, (32, 128), dtype=torch.int64)
        packed = pack_indices(indices, bits=bits)
        assert packed.dtype == torch.uint8
        assert packed.shape == (32, expected_cols)
        assert expected_cols == packed_group_bytes(bits, 128)
        unpacked = unpack_indices(packed, bits, 128)
        assert torch.equal(indices, unpacked)

    def test_all_3bit_codes_survive_packing(self):
        # Cycle through every code at every position within the 8-value
        # packing period, including the two byte-crossing codes.
        indices = torch.arange(8 * 128, dtype=torch.int64).reshape(8, 128) % 8
        packed = pack_indices(indices, bits=3)
        assert torch.equal(unpack_indices(packed, 3, 128), indices)


class TestWrapper:
    """TurboQuantWrapper drop-in replacement (reference CPU path)."""

    def test_wrapper_forward_close_to_original(self):
        torch.manual_seed(42)
        linear = torch.nn.Linear(512, 256, bias=True)
        x = torch.randn(4, 512)
        with torch.no_grad():
            y_orig = linear(x)
            y_wrapped = TurboQuantWrapper(linear, bits=4)(x)
        rel_error = ((y_orig - y_wrapped) ** 2).mean() / (y_orig**2).mean()
        assert rel_error < 0.05, f"Wrapper relative error {rel_error:.4f} too high"

    def test_wrapper_no_bias(self):
        linear = torch.nn.Linear(256, 128, bias=False)
        wrapper = TurboQuantWrapper(linear, bits=3)
        x = torch.randn(2, 256)
        with torch.no_grad():
            y = wrapper(x)
        assert y.shape == (2, 128)

    def test_wrapper_memory_smaller(self):
        linear = torch.nn.Linear(1024, 1024, bias=False)
        orig_bytes = linear.weight.numel() * linear.weight.element_size()
        wrapper = TurboQuantWrapper(linear, bits=3)
        comp_bytes = wrapper.packed_weight.numel() + wrapper.norms.numel() * wrapper.norms.element_size()
        assert comp_bytes < orig_bytes

    def test_wrapper_batch_sizes(self):
        linear = torch.nn.Linear(512, 256, bias=True)
        wrapper = TurboQuantWrapper(linear, bits=4)
        for batch in [1, 4, 16]:
            with torch.no_grad():
                y = wrapper(torch.randn(batch, 512))
            assert y.shape == (batch, 256)

    def test_wrapper_pads_in_features(self):
        # in_features not a multiple of group_size exercises the padding path
        linear = torch.nn.Linear(96, 64, bias=False)
        wrapper = TurboQuantWrapper(linear, bits=4, group_size=64)
        with torch.no_grad():
            y = wrapper(torch.randn(2, 96))
        assert y.shape == (2, 64)


class TestLearnedRotation:
    """Regression: learned-rotation weights must dequantize with the stored
    rotation, not the fixed WHT (which produced garbage output)."""

    @staticmethod
    def _random_orthogonal(n: int, seed: int = 0) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen))
        return q

    def test_learned_rotation_forward_matches_original(self):
        torch.manual_seed(42)
        group_size = 128
        linear = torch.nn.Linear(256, 128, bias=True)
        rotation = self._random_orthogonal(group_size)

        x = torch.randn(4, 256)
        with torch.no_grad():
            y_orig = linear(x)
            y_wrapped = TurboQuantWrapper(linear, bits=4, group_size=group_size, rotation=rotation)(x)

        rel_error = ((y_orig - y_wrapped) ** 2).mean() / (y_orig**2).mean()
        # Before the fix the WHT was applied to R-rotated codes and the
        # relative error was O(1) — the outputs were unrelated.
        assert rel_error < 0.05, f"Learned-rotation relative error {rel_error:.4f} too high"

    def test_learned_rotation_matches_manual_dequant(self):
        torch.manual_seed(1)
        group_size = 64
        linear = torch.nn.Linear(128, 32, bias=False)
        rotation = self._random_orthogonal(group_size, seed=3)
        wrapper = TurboQuantWrapper(linear, bits=3, group_size=group_size, rotation=rotation)

        indices = unpack_indices(wrapper.packed_weight, 3, group_size)
        y_hat = wrapper.tq_centroids.to(torch.float32)[indices]
        w_manual = (y_hat @ rotation) * wrapper.norms.reshape(-1, 1)
        w_manual = w_manual.reshape(32, 128)

        x = torch.randn(2, 128)
        with torch.no_grad():
            y_wrapped = wrapper(x)
        torch.testing.assert_close(y_wrapped, x @ w_manual.t().to(x.dtype), rtol=1e-3, atol=1e-4)

    def test_learned_rotation_norms_are_fp32(self):
        linear = torch.nn.Linear(128, 32, bias=False)
        rotation = self._random_orthogonal(128, seed=5)
        wrapper = TurboQuantWrapper(linear, bits=3, rotation=rotation)
        assert wrapper.norms.dtype == torch.float32


class TestSelectBits:
    """Sensitive-layer bit selection used by checkpoints and the vLLM path."""

    def test_default_patterns(self):
        assert select_bits("model.layers.0.self_attn.o_proj.weight", 3, 4) == 4
        assert select_bits("model.layers.0.mlp.down_proj.weight", 3, 4) == 4
        assert select_bits("model.layers.0.mlp.gate_proj.weight", 3, 4) == 3

    def test_no_sensitive_bits(self):
        assert select_bits("model.layers.0.self_attn.o_proj.weight", 3, None) == 3

    def test_custom_patterns(self):
        assert select_bits("x.q_proj.weight", 3, 4, sensitive_patterns=("q_proj",)) == 4
        assert select_bits("x.o_proj.weight", 3, 4, sensitive_patterns=("q_proj",)) == 3

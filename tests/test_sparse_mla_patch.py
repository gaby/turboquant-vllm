"""Tests for the sparse-MLA patch path in turboquant_vllm.vllm_patch.

vllm-project/vllm#38476 ships SparseMLAAttentionImpl as a sibling of
MLAAttentionImpl, so the base-class MLACommonImpl patch does not reach it.
These tests lock in that the concrete class gets patched directly, that a
decode-only impl (forward_mqa but no forward_mha) is handled, and that
resolution degrades gracefully when the PR is not installed (the common case).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPatchMLAImpl(unittest.TestCase):
    def test_patches_cache_update_and_forward(self):
        from turboquant_vllm import vllm_patch

        class FakeSparseMLAImpl:
            def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
                return "orig"

            def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata):
                return "orig"

        backends = []
        vllm_patch._patch_mla_impl(FakeSparseMLAImpl, ("forward_mha", "forward_mqa"), "SparseMLA", backends)

        self.assertEqual(backends, ["SparseMLA"])
        # functools.wraps sets __wrapped__ on the replacement, which is the
        # observable signal that the method was wrapped rather than left intact.
        self.assertTrue(hasattr(FakeSparseMLAImpl.do_kv_cache_update, "__wrapped__"))
        self.assertTrue(hasattr(FakeSparseMLAImpl.forward_mqa, "__wrapped__"))

    def test_decode_only_impl_skips_missing_forward_mha(self):
        # A sparse impl that defines only forward_mqa must be patched without
        # error; the absent forward_mha must be skipped, not synthesized.
        from turboquant_vllm import vllm_patch

        class OnlyMqa:
            def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
                return None

            def forward_mqa(self, q, kv_cache, attn_metadata):
                return None

        backends = []
        vllm_patch._patch_mla_impl(OnlyMqa, ("forward_mha", "forward_mqa"), "SparseMLA", backends)

        self.assertEqual(backends, ["SparseMLA"])
        self.assertFalse(hasattr(OnlyMqa, "forward_mha"))
        self.assertTrue(hasattr(OnlyMqa.forward_mqa, "__wrapped__"))


class TestResolveSparseMLA(unittest.TestCase):
    def setUp(self):
        from turboquant_vllm import vllm_patch

        self.vllm_patch = vllm_patch
        self._orig = vllm_patch._SPARSE_MLA_CANDIDATES

    def tearDown(self):
        self.vllm_patch._SPARSE_MLA_CANDIDATES = self._orig

    def test_returns_none_when_absent(self):
        # The common case: vllm #38476 is not installed, so no candidate
        # resolves and the resolver returns None instead of raising.
        self.vllm_patch._SPARSE_MLA_CANDIDATES = (("turboquant_vllm._no_such_module", "Nope"),)
        self.assertIsNone(self.vllm_patch._resolve_sparse_mla_impl())

    def test_resolves_first_available_candidate(self):
        # First candidate missing, second resolvable: the walker must skip the
        # ImportError and return the class from the next candidate.
        from collections import OrderedDict

        self.vllm_patch._SPARSE_MLA_CANDIDATES = (
            ("turboquant_vllm._no_such_module", "Nope"),
            ("collections", "OrderedDict"),
        )
        self.assertIs(self.vllm_patch._resolve_sparse_mla_impl(), OrderedDict)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from turboquant_vllm.weight_quant import dtype_to_config_name, resolve_torch_dtype


class TestDtypeResolution(unittest.TestCase):
    def test_resolve_common_aliases(self):
        self.assertEqual(resolve_torch_dtype("float16"), torch.float16)
        self.assertEqual(resolve_torch_dtype("half"), torch.float16)
        self.assertEqual(resolve_torch_dtype("bfloat16"), torch.bfloat16)

    def test_resolve_fp8_alias_when_available(self):
        resolved = resolve_torch_dtype("fp8")
        if hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz") or hasattr(torch, "float8_e5m2"):
            self.assertIsNotNone(resolved)
            self.assertIn("float8", str(resolved))
        else:
            self.assertIsNone(resolved)

    def test_dtype_to_config_name(self):
        self.assertEqual(dtype_to_config_name(torch.float16), "float16")
        self.assertEqual(dtype_to_config_name("bfloat16"), "bfloat16")
        self.assertEqual(dtype_to_config_name("unknown_dtype"), "float16")


if __name__ == "__main__":
    unittest.main(verbosity=2)

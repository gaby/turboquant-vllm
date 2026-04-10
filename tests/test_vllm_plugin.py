import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def test_register_native_backend_preserves_classmethod_semantics(monkeypatch):
    registry_calls = []

    registry_module = types.ModuleType("vllm.v1.attention.backends.registry")

    class AttentionBackendEnum:
        CUSTOM = "custom"

    def register_backend(*args):
        registry_calls.append(args)

    registry_module.AttentionBackendEnum = AttentionBackendEnum
    registry_module.register_backend = register_backend

    platforms_cuda_module = types.ModuleType("vllm.platforms.cuda")

    class FakeCudaPlatform:
        original_calls = []

        @classmethod
        def get_valid_backends(cls, device_capability, attn_selector_config, num_heads=None):
            cls.original_calls.append((cls, device_capability, attn_selector_config, num_heads))
            return [("original", num_heads)], {"source": "original"}

    platforms_cuda_module.CudaPlatform = FakeCudaPlatform

    native_backend_module = types.ModuleType("turboquant_vllm.native_backend")

    class TurboQuantAttentionBackend:
        pass

    TurboQuantAttentionBackend.__module__ = "turboquant_vllm.native_backend"
    native_backend_module.TurboQuantAttentionBackend = TurboQuantAttentionBackend

    stub_modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.attention": types.ModuleType("vllm.v1.attention"),
        "vllm.v1.attention.backends": types.ModuleType("vllm.v1.attention.backends"),
        "vllm.v1.attention.backends.registry": registry_module,
        "vllm.platforms": types.ModuleType("vllm.platforms"),
        "vllm.platforms.cuda": platforms_cuda_module,
        "turboquant_vllm.native_backend": native_backend_module,
    }

    for name, module in stub_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    plugin_path = Path(__file__).resolve().parents[1] / "turboquant_vllm" / "_vllm_plugin.py"
    spec = importlib.util.spec_from_file_location("test_vllm_plugin_module", plugin_path)
    assert spec is not None and spec.loader is not None
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)

    assert plugin._register_native_backend() is True
    assert registry_calls == [
        (
            AttentionBackendEnum.CUSTOM,
            "turboquant_vllm.native_backend.TurboQuantAttentionBackend",
        )
    ]
    assert isinstance(FakeCudaPlatform.__dict__["get_valid_backends"], classmethod)

    tq_backends, tq_meta = FakeCudaPlatform.get_valid_backends(
        "sm90",
        SimpleNamespace(kv_cache_dtype="tq3"),
        8,
    )
    assert tq_backends == [(AttentionBackendEnum.CUSTOM, 0)]
    assert tq_meta == {}

    default_backends, default_meta = FakeCudaPlatform.get_valid_backends(
        "sm90",
        SimpleNamespace(kv_cache_dtype="auto"),
        4,
    )
    assert default_backends == [("original", 4)]
    assert default_meta == {"source": "original"}
    assert FakeCudaPlatform.original_calls == [
        (FakeCudaPlatform, "sm90", SimpleNamespace(kv_cache_dtype="auto"), 4)
    ]

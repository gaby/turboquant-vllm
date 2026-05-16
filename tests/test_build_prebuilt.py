import json

from turboquant_vllm import build as tq_build


def test_arches_from_gencode_flags():
    flags = [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_121,code=sm_121",
        "-O3",
    ]

    assert tq_build._arches_from_gencode_flags(flags) == ["80", "121"]


def test_prebuilt_without_manifest_is_allowed_when_no_cuda_visible(monkeypatch, tmp_path):
    so = tmp_path / "turbo_quant_cuda.so"
    so.touch()
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: [])

    assert tq_build._prebuilt_is_compatible(so) is True


def test_prebuilt_without_manifest_is_rejected_on_cuda_host(monkeypatch, tmp_path):
    so = tmp_path / "turbo_quant_cuda.so"
    so.touch()
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["121"])
    monkeypatch.delenv("TQ_CUDA_ALLOW_UNVERIFIED_PREBUILT", raising=False)

    assert tq_build._prebuilt_is_compatible(so) is False


def test_unverified_prebuilt_can_be_explicitly_allowed(monkeypatch, tmp_path):
    so = tmp_path / "turbo_quant_cuda.so"
    so.touch()
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["121"])
    monkeypatch.setenv("TQ_CUDA_ALLOW_UNVERIFIED_PREBUILT", "1")

    assert tq_build._prebuilt_is_compatible(so) is True


def test_prebuilt_manifest_must_cover_local_arches(monkeypatch, tmp_path):
    so = tmp_path / "turbo_quant_cuda.so"
    so.touch()
    tq_build._prebuilt_manifest_path(so).write_text(json.dumps({"arches": ["80", "90"]}))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["121"])

    assert tq_build._prebuilt_is_compatible(so) is False


def test_prebuilt_manifest_accepts_matching_local_arch(monkeypatch, tmp_path):
    so = tmp_path / "turbo_quant_cuda.so"
    so.touch()
    tq_build._prebuilt_manifest_path(so).write_text(json.dumps({"arches": ["80", "121"]}))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["121"])

    assert tq_build._prebuilt_is_compatible(so) is True


def test_force_jit_skips_prebuilt_loader(monkeypatch, tmp_path):
    so = tmp_path / "turbo_quant_cuda.so"
    so.touch()
    monkeypatch.setenv("TQ_CUDA_FORCE_JIT", "1")
    monkeypatch.setattr(tq_build, "_candidate_prebuilt_paths", lambda: [so])

    def fail_load(_path):
        raise AssertionError("force-JIT should not attempt to load prebuilt modules")

    monkeypatch.setattr(tq_build, "_load_module_from_path", fail_load)

    assert tq_build._load_prebuilt_module() is None

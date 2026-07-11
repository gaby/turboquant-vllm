import json

import pytest

from turboquant_vllm import build as tq_build


@pytest.fixture
def no_arch_override(monkeypatch):
    monkeypatch.delenv("TQ_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)


def test_arches_from_gencode_flags():
    flags = [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_121,code=sm_121",
        "-gencode=arch=compute_90,code=compute_90",
        "-O3",
    ]

    assert tq_build._arches_from_gencode_flags(flags) == ["80", "121"]


def test_ptx_arches_from_gencode_flags():
    flags = [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=compute_90",
        "-O3",
    ]

    assert tq_build._ptx_arches_from_gencode_flags(flags) == ["90"]


def test_gencode_local_b200_on_old_toolkit_falls_back_to_ptx(monkeypatch, no_arch_override):
    """CUDA 12.4 cannot compile sm_100 (B200); expect compute_90 PTX instead of a failed build."""
    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (12, 4))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["100"])

    flags = tq_build._gencode_flags()

    assert flags == ["-gencode=arch=compute_90,code=compute_90"]


def test_gencode_local_b200_on_cuda_128(monkeypatch, no_arch_override):
    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (12, 8))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["100"])

    assert tq_build._gencode_flags() == ["-gencode=arch=compute_100,code=sm_100"]


def test_gencode_local_b300_gating(monkeypatch, no_arch_override):
    """sm_103 (B300) needs CUDA 12.9; on 12.8 the best PTX base is compute_100."""
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["103"])

    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (12, 8))
    assert tq_build._gencode_flags() == ["-gencode=arch=compute_100,code=compute_100"]

    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (12, 9))
    assert tq_build._gencode_flags() == ["-gencode=arch=compute_103,code=sm_103"]


def test_gencode_cuda13_orders_after_cuda129(monkeypatch, no_arch_override):
    """(13, 0) must compare greater than (12, 9) — sm_121 stays compilable on CUDA 13."""
    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (13, 0))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["121"])

    assert tq_build._gencode_flags() == ["-gencode=arch=compute_121,code=sm_121"]


def test_gencode_cuda13_drops_pre_turing(monkeypatch, no_arch_override):
    """CUDA 13 removed sm_70; H100 in the same box still gets native SASS."""
    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (13, 0))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["70", "90"])

    flags = tq_build._gencode_flags()

    assert "-gencode=arch=compute_90,code=sm_90" in flags
    assert not any("compute_70" in f for f in flags)


def test_gencode_override_skips_uncompilable_arch(monkeypatch, no_arch_override):
    monkeypatch.setenv("TQ_CUDA_ARCH_LIST", "90,100")
    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (12, 4))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: [])

    assert tq_build._gencode_flags() == ["-gencode=arch=compute_90,code=sm_90"]


def test_gencode_override_accepts_torch_space_separated_format(monkeypatch, no_arch_override):
    """TORCH_CUDA_ARCH_LIST conventionally uses spaces and '+PTX' suffixes."""
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0 9.0+PTX")
    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (12, 4))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: [])

    assert tq_build._gencode_flags() == [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_90,code=compute_90",
    ]


def test_gencode_override_accepts_named_and_suffixed_arches(monkeypatch, no_arch_override):
    """PyTorch also accepts named arches ('Hopper') and feature-suffixed
    numeric arches ('9.0a', '12.0f'); both must not be silently dropped."""
    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (12, 9))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: [])

    # Named arches imply '+PTX' on their highest target, matching PyTorch's
    # own expansions (Hopper == '9.0+PTX', Ampere == '8.0;8.6+PTX').
    for token in ("Hopper", "Hopper+PTX"):
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", token)
        assert tq_build._gencode_flags() == [
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90,code=compute_90",
        ]

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "Ampere")
    assert tq_build._gencode_flags() == [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_86,code=compute_86",
    ]

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0a 12.0f")
    assert tq_build._gencode_flags() == [
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_120,code=sm_120",
    ]


def test_gencode_fallback_list_per_cuda_version(monkeypatch, no_arch_override):
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: [])

    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (12, 4))
    flags = tq_build._gencode_flags()
    assert tq_build._arches_from_gencode_flags(flags) == ["80", "86", "89", "90"]
    assert tq_build._ptx_arches_from_gencode_flags(flags) == ["90"]

    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (12, 8))
    flags = tq_build._gencode_flags()
    assert tq_build._arches_from_gencode_flags(flags) == ["80", "86", "89", "90", "100", "120"]

    for version in ((12, 9), (13, 0)):
        monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda v=version: v)
        flags = tq_build._gencode_flags()
        assert tq_build._arches_from_gencode_flags(flags) == ["80", "86", "89", "90", "100", "103", "120", "121"]
        assert tq_build._ptx_arches_from_gencode_flags(flags) == ["90"]


def test_gencode_fallback_unknown_cuda_stays_conservative(monkeypatch, no_arch_override):
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: [])
    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (0, 0))

    flags = tq_build._gencode_flags()

    assert tq_build._arches_from_gencode_flags(flags) == ["80", "86", "89", "90"]
    assert tq_build._ptx_arches_from_gencode_flags(flags) == ["80"]


def test_gencode_fallback_gates_base_arches_on_old_toolkit(monkeypatch, no_arch_override):
    """CUDA 11.0 cannot compile compute_89/90 — the fallback must degrade, not fail nvcc."""
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: [])
    monkeypatch.setattr(tq_build, "_cuda_version_tuple", lambda: (11, 0))

    flags = tq_build._gencode_flags()

    assert tq_build._arches_from_gencode_flags(flags) == ["80"]
    assert tq_build._ptx_arches_from_gencode_flags(flags) == ["80"]


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


def test_prebuilt_manifest_ptx_covers_newer_local_arch(monkeypatch, tmp_path):
    """compute_90 PTX is driver-JIT-able on sm_100 (B200) — accept the prebuilt."""
    so = tmp_path / "turbo_quant_cuda.so"
    so.touch()
    tq_build._prebuilt_manifest_path(so).write_text(json.dumps({"arches": ["80", "90"], "ptx": ["90"]}))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["100"])

    assert tq_build._prebuilt_is_compatible(so) is True


def test_prebuilt_manifest_ptx_does_not_cover_older_local_arch(monkeypatch, tmp_path):
    """PTX only JIT-compiles forward: compute_90 PTX cannot run on sm_86."""
    so = tmp_path / "turbo_quant_cuda.so"
    so.touch()
    tq_build._prebuilt_manifest_path(so).write_text(json.dumps({"arches": ["90"], "ptx": ["90"]}))
    monkeypatch.setattr(tq_build, "_detect_local_arches", lambda: ["86"])
    monkeypatch.delenv("TQ_CUDA_ALLOW_UNVERIFIED_PREBUILT", raising=False)

    assert tq_build._prebuilt_is_compatible(so) is False


def test_force_jit_skips_prebuilt_loader(monkeypatch, tmp_path):
    so = tmp_path / "turbo_quant_cuda.so"
    so.touch()
    monkeypatch.setenv("TQ_CUDA_FORCE_JIT", "1")
    monkeypatch.setattr(tq_build, "_candidate_prebuilt_paths", lambda: [so])

    def fail_load(_path):
        raise AssertionError("force-JIT should not attempt to load prebuilt modules")

    monkeypatch.setattr(tq_build, "_load_module_from_path", fail_load)

    assert tq_build._load_prebuilt_module() is None

"""Build the TurboQuant+ CUDA extension.

Usage:
    python -m turboquant_vllm.build

This compiles csrc/turbo_quant.cu and csrc/torch_bindings.cpp into
a shared library that can be loaded as a PyTorch extension.
"""

import logging
import json
import os
import shutil
from importlib import util as importlib_util
from pathlib import Path

logger = logging.getLogger(__name__)

# csrc/ is either a sibling of turboquant_vllm/ (dev) or a sibling package (installed)
_pkg_dir = Path(__file__).resolve().parent
CSRC_DIR = _pkg_dir.parent / "csrc"
if not (CSRC_DIR / "turbo_quant.cu").exists():
    # Installed as package — csrc is a sibling package in site-packages
    CSRC_DIR = _pkg_dir.parent / "csrc"
if not (CSRC_DIR / "turbo_quant.cu").exists():
    raise FileNotFoundError(
        f"Cannot find csrc/turbo_quant.cu. Searched: {_pkg_dir.parent / 'csrc'}. "
        "Install from source (git clone) to get CUDA kernels, or use PyTorch fallback."
    )

PREBUILT_DIR = _pkg_dir / "_native"
PREBUILT_BASENAME = "turbo_quant_cuda"
PREBUILT_MANIFEST_SUFFIX = ".arches.json"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _cuda_version_tuple():
    """Return (major, minor) for the CUDA toolkit torch was built against.

    Matches the toolchain `torch.utils.cpp_extension.load()` will invoke,
    so it's the right version to gate gencode flags on.
    """
    import torch

    v = getattr(torch.version, "cuda", None) or "0.0"
    parts = v.split(".")
    try:
        major = int(parts[0])
    except (TypeError, ValueError):
        return (0, 0)
    try:
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        minor = 0
    return (major, minor)


def _detect_local_arches() -> list[str]:
    """Return sorted local CUDA SM targets like ``["86"]``.

    Compiling for every historical architecture makes the first JIT build
    expensive in both time and host RAM. For runtime JIT we default to the
    local machine's visible GPUs instead. Users can override this with the
    standard ``TORCH_CUDA_ARCH_LIST`` env var or ``TQ_CUDA_ARCH_LIST``.
    """
    import torch

    if not torch.cuda.is_available():
        return []
    arches = {
        f"{major}{minor}"
        for idx in range(torch.cuda.device_count())
        for major, minor in [torch.cuda.get_device_capability(idx)]
    }
    return sorted(arches, key=int)


def _gencode_flags() -> list[str]:
    """Build a compact gencode list for the current runtime host."""
    cuda_major, cuda_minor = _cuda_version_tuple()

    override = os.environ.get("TQ_CUDA_ARCH_LIST") or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if override:
        arch_tokens = [tok.strip() for tok in override.replace(";", ",").split(",") if tok.strip()]
        flags: list[str] = []
        for token in arch_tokens:
            token = token.replace(".", "")
            if not token.isdigit():
                continue
            arch_num = int(token)
            if arch_num == 121 and (cuda_major, cuda_minor) < (12, 9):
                continue
            flags.append(f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}")
        if flags:
            return flags

    local_arches = _detect_local_arches()
    if local_arches:
        flags = []
        for arch in local_arches:
            arch_num = int(arch)
            if arch_num == 121 and (cuda_major, cuda_minor) < (12, 9):
                continue
            flags.append(f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}")
        if flags:
            return flags

    # Fallback for environments where no GPU is visible during build.
    flags = [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]
    if (cuda_major, cuda_minor) >= (12, 9):
        flags.append("-gencode=arch=compute_121,code=sm_121")
    return flags


def _arches_from_gencode_flags(flags: list[str]) -> list[str]:
    """Extract SM targets like ``["80", "121"]`` from nvcc gencode flags."""
    arches: set[str] = set()
    for flag in flags:
        marker = "code=sm_"
        if marker not in flag:
            continue
        arch = flag.split(marker, 1)[1].split(",", 1)[0].strip()
        if arch.isdigit():
            arches.add(arch)
    return sorted(arches, key=int)


def _candidate_prebuilt_paths() -> list[Path]:
    """Return candidate prebuilt extension paths in priority order."""
    candidates: list[Path] = []
    explicit = os.environ.get("TQ_CUDA_PREBUILT_PATH")
    if explicit:
        candidates.append(Path(explicit))

    for directory in (PREBUILT_DIR, _pkg_dir):
        if directory.exists():
            candidates.extend(sorted(directory.glob(f"{PREBUILT_BASENAME}*.so")))

    return candidates


def _prebuilt_manifest_path(path: Path) -> Path:
    return path.with_name(path.name + PREBUILT_MANIFEST_SUFFIX)


def _read_prebuilt_arches(path: Path) -> set[str] | None:
    manifest = _prebuilt_manifest_path(path)
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read TurboQuant prebuilt manifest %s: %s", manifest, exc)
        return None

    arches = data.get("arches")
    if not isinstance(arches, list):
        logger.warning("TurboQuant prebuilt manifest %s has no arches list", manifest)
        return None
    parsed = {str(arch).replace(".", "") for arch in arches}
    return {arch for arch in parsed if arch.isdigit()}


def _prebuilt_is_compatible(path: Path) -> bool:
    """Return whether ``path`` is safe to use on this host.

    A prebuilt extension can import successfully even when it was not built
    for the local GPU. On new architectures (for example GB10 / sm_121),
    silently accepting that path makes performance diagnosis unreliable.
    Bundled builds therefore carry a tiny sidecar manifest with the SM
    targets compiled into the shared object. If a CUDA device is visible, the
    prebuilt must prove it covers every local SM unless the user explicitly
    opts into unverified loading.
    """
    local_arches = set(_detect_local_arches())
    if not local_arches:
        return True

    if _env_flag("TQ_CUDA_ALLOW_UNVERIFIED_PREBUILT"):
        logger.warning(
            "TQ_CUDA_ALLOW_UNVERIFIED_PREBUILT=1: using %s without SM coverage verification "
            "for local arches %s",
            path,
            sorted(local_arches, key=int),
        )
        return True

    prebuilt_arches = _read_prebuilt_arches(path)
    if prebuilt_arches is None:
        logger.warning(
            "Skipping unverified TurboQuant prebuilt extension %s on CUDA arches %s; "
            "falling back to local JIT. Set TQ_CUDA_ALLOW_UNVERIFIED_PREBUILT=1 to override.",
            path,
            sorted(local_arches, key=int),
        )
        return False

    missing = local_arches - prebuilt_arches
    if missing:
        logger.warning(
            "Skipping TurboQuant prebuilt extension %s: manifest arches %s do not cover "
            "local CUDA arches %s; missing %s. Falling back to local JIT.",
            path,
            sorted(prebuilt_arches, key=int),
            sorted(local_arches, key=int),
            sorted(missing, key=int),
        )
        return False
    return True


def _load_module_from_path(path: Path):
    spec = importlib_util.spec_from_file_location(PREBUILT_BASENAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prebuilt_module():
    """Load a prebuilt extension bundled into the package/image."""
    if _env_flag("TQ_CUDA_FORCE_JIT") or _env_flag("TQ_CUDA_DISABLE_PREBUILT"):
        logger.warning("Skipping TurboQuant prebuilt CUDA extension because force-JIT/disable-prebuilt is set")
        return None

    for candidate in _candidate_prebuilt_paths():
        if not candidate.is_file():
            continue
        if not _prebuilt_is_compatible(candidate):
            continue
        try:
            module = _load_module_from_path(candidate)
            logger.warning(
                "Loaded prebuilt TurboQuant CUDA extension from %s (manifest arches=%s)",
                candidate,
                sorted(_read_prebuilt_arches(candidate) or [], key=int),
            )
            return module
        except Exception as exc:
            logger.warning("Failed to load prebuilt TurboQuant CUDA extension from %s: %s", candidate, exc)
    return None


def _bundle_module(module, arches: list[str]) -> Path:
    """Copy the compiled extension into the package for runtime reuse."""
    PREBUILT_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(module.__file__).resolve()
    target = PREBUILT_DIR / source.name
    if source != target:
        shutil.copy2(source, target)
    _prebuilt_manifest_path(target).write_text(json.dumps({"arches": arches}, indent=2) + "\n")
    logger.warning("Bundled TurboQuant CUDA extension to %s", target)
    return target


def build():
    """JIT-compile the CUDA extension. Returns the loaded module."""
    from torch.utils.cpp_extension import load

    prebuilt = _load_prebuilt_module()
    if prebuilt is not None:
        return prebuilt

    # Runtime JIT compilation can otherwise fan out to many ninja workers
    # and transiently consume tens of GiB of host RAM. Keep the default
    # conservative; power users can override via MAX_JOBS.
    os.environ.setdefault("MAX_JOBS", os.environ.get("TQ_CUDA_MAX_JOBS", "1"))
    gencode_flags = _gencode_flags()

    logger.warning(
        "TurboQuant CUDA build config: MAX_JOBS=%s TQ_CUDA_MAX_JOBS=%s "
        "TQ_CUDA_ARCH_LIST=%s TORCH_CUDA_ARCH_LIST=%s final_gencode=%s",
        os.environ.get("MAX_JOBS"),
        os.environ.get("TQ_CUDA_MAX_JOBS"),
        os.environ.get("TQ_CUDA_ARCH_LIST"),
        os.environ.get("TORCH_CUDA_ARCH_LIST"),
        " ".join(gencode_flags),
    )

    sources = [
        str(CSRC_DIR / "turbo_quant.cu"),
        str(CSRC_DIR / "tq_weight_dequant.cu"),
        str(CSRC_DIR / "tq_weight_gemv_bs1.cu"),
        str(CSRC_DIR / "torch_bindings.cpp"),
    ]

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        *gencode_flags,
    ]

    module = load(
        name="turbo_quant_cuda",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[str(CSRC_DIR)],
        verbose=True,
    )
    if os.environ.get("TQ_CUDA_BUNDLE", "0") == "1":
        _bundle_module(module, _arches_from_gencode_flags(gencode_flags))
    return module


if __name__ == "__main__":
    os.environ.setdefault("TQ_CUDA_BUNDLE", "1")
    mod = build()
    print(f"Built successfully: {mod}")
    print(f"Available functions: {[x for x in dir(mod) if not x.startswith('_')]}")

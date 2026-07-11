"""Build the TurboQuant+ CUDA extension.

Usage:
    python -m turboquant_vllm.build

This compiles csrc/turbo_quant.cu and csrc/torch_bindings.cpp into
a shared library that can be loaded as a PyTorch extension.
"""

import logging
import json
import os
import re
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


# nvcc toolkit version that introduced each SM target. Datacenter Blackwell
# (B200 = sm_100, B300/GB300 = sm_103) and consumer/DGX-Spark Blackwell
# (sm_120 / sm_121) need CUDA 12.8/12.9; Thor (sm_110) needs CUDA 13.0.
_ARCH_MIN_CUDA: dict[int, tuple[int, int]] = {
    70: (9, 0),
    72: (9, 0),
    75: (10, 0),
    80: (11, 0),
    86: (11, 1),
    87: (11, 4),
    89: (11, 8),
    90: (11, 8),
    100: (12, 8),
    101: (12, 8),
    103: (12, 9),
    110: (13, 0),
    120: (12, 8),
    121: (12, 9),
}

# Newest-first candidates for a virtual (PTX) target when the toolkit cannot
# emit SASS for the local GPU. PTX is forward-JIT-compiled by the driver, so
# e.g. compute_90 PTX runs on sm_100/sm_103 hardware under a CUDA 12.4 torch.
# Embedded/Tegra targets (72, 87, 101, 110, 121) are excluded — their PTX is
# not a sensible base for discrete GPUs.
_PTX_FALLBACK_CANDIDATES = (120, 103, 100, 90, 89, 86, 80, 75, 70)

# CUDA 13.0 removed every pre-Turing target; sm_75 is that toolchain's
# minimum. Bump this (or generalize to a per-arch removal map) when a future
# major CUDA release drops more targets.
_CUDA13_MIN_ARCH = 75

# Named architectures accepted by PyTorch's TORCH_CUDA_ARCH_LIST (see
# torch.utils.cpp_extension._get_cuda_arch_flags) mapped to the SM targets
# they expand to for discrete GPUs. PyTorch's named expansions always carry
# an implicit '+PTX' on their highest target (e.g. Ampere -> '8.0;8.6+PTX'),
# which the parser below mirrors.
_NAMED_ARCH_TOKENS: dict[str, tuple[int, ...]] = {
    "TURING": (75,),
    "AMPERE": (80, 86),
    "ADA": (89,),
    "HOPPER": (90,),
    "BLACKWELL": (100, 120),
}

# "9.0a" / "12.0f" style tokens: the trailing letter selects an
# arch-conditional (a) or family (f) feature set. Our kernels use neither,
# so the base SM target is the right translation.
_ARCH_TOKEN_RE = re.compile(r"^(\d+)[AF]?$")


def _arch_supported_by_cuda(arch_num: int, cuda_version: tuple[int, int]) -> bool:
    """Whether the nvcc toolkit at ``cuda_version`` can compile ``arch_num``.

    An unknown toolkit version ``(0, 0)`` returns True — don't second-guess
    explicit user requests when torch reports no CUDA version.
    """
    if cuda_version <= (0, 0):
        return True
    if cuda_version < _ARCH_MIN_CUDA.get(arch_num, (0, 0)):
        return False
    if cuda_version >= (13, 0) and arch_num < _CUDA13_MIN_ARCH:
        return False
    return True


def _ptx_fallback_arch(arch_num: int, cuda_version: tuple[int, int]) -> int | None:
    """Best virtual (PTX) target ≤ ``arch_num`` the toolkit can emit."""
    for candidate in _PTX_FALLBACK_CANDIDATES:
        if candidate <= arch_num and _arch_supported_by_cuda(candidate, cuda_version):
            return candidate
    return None


def _sass_flag(arch_num: int) -> str:
    return f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}"


def _ptx_flag(arch_num: int) -> str:
    return f"-gencode=arch=compute_{arch_num},code=compute_{arch_num}"


def _gencode_flags() -> list[str]:
    """Build a compact gencode list for the current runtime host."""
    cuda_version = _cuda_version_tuple()

    override = os.environ.get("TQ_CUDA_ARCH_LIST") or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if override:
        # TORCH_CUDA_ARCH_LIST conventionally separates entries with spaces
        # ("8.0 9.0+PTX", "Hopper", "9.0a"), but commas/semicolons are also
        # seen in the wild.
        arch_tokens = [tok for tok in re.split(r"[,;\s]+", override) if tok]
        flags: list[str] = []
        for token in arch_tokens:
            wants_ptx = token.upper().endswith("+PTX")
            if wants_ptx:
                token = token[: -len("+PTX")]
            named = _NAMED_ARCH_TOKENS.get(token.upper())
            if named is not None:
                arch_nums = list(named)
                # Mirror PyTorch: named arches imply PTX on their highest
                # SM target ('Hopper' == '9.0+PTX').
                wants_ptx = True
            else:
                match = _ARCH_TOKEN_RE.match(token.replace(".", "").upper())
                if match is None:
                    logger.warning("Ignoring unrecognized CUDA arch token %r in arch-list override", token)
                    continue
                arch_nums = [int(match.group(1))]
            supported = [a for a in arch_nums if _arch_supported_by_cuda(a, cuda_version)]
            for arch_num in arch_nums:
                if arch_num not in supported:
                    logger.warning(
                        "Skipping requested CUDA arch sm_%d: not compilable by CUDA %d.%d",
                        arch_num,
                        *cuda_version,
                    )
            for arch_num in supported:
                flags.append(_sass_flag(arch_num))
                if wants_ptx and arch_num == max(supported):
                    flags.append(_ptx_flag(arch_num))
        if flags:
            return flags

    local_arches = _detect_local_arches()
    if local_arches:
        flags = []
        ptx_arches: set[int] = set()
        for arch in local_arches:
            arch_num = int(arch)
            if _arch_supported_by_cuda(arch_num, cuda_version):
                flags.append(_sass_flag(arch_num))
                continue
            # Toolkit too old for this GPU (e.g. B200/sm_100 or B300/sm_103
            # under CUDA < 12.8/12.9). Emit the newest PTX the toolkit knows
            # so the driver can JIT for the local arch, instead of failing
            # the whole nvcc invocation on an unknown compute_XXX.
            ptx_arch = _ptx_fallback_arch(arch_num, cuda_version)
            if ptx_arch is None:
                logger.warning(
                    "Local CUDA arch sm_%d is not compilable by CUDA %d.%d and no PTX fallback exists",
                    arch_num,
                    *cuda_version,
                )
                continue
            logger.warning(
                "Local CUDA arch sm_%d is not compilable by CUDA %d.%d; "
                "emitting compute_%d PTX for driver-side JIT instead. "
                "Upgrade to a CUDA >= %d.%d torch build for native SASS.",
                arch_num,
                *cuda_version,
                ptx_arch,
                *_ARCH_MIN_CUDA.get(arch_num, (0, 0)),
            )
            ptx_arches.add(ptx_arch)
        flags.extend(_ptx_flag(a) for a in sorted(ptx_arches))
        if flags:
            return flags

    # Fallback for environments where no GPU is visible during build: cover
    # the common datacenter/workstation SASS targets the toolkit supports,
    # plus one PTX target for forward compatibility with newer arches.
    # With an unknown toolkit version, stay conservative: pre-Blackwell SASS
    # only (matching the historical list), since we can't prove newer
    # compute_XXX targets are compilable.
    known_version = cuda_version > (0, 0)
    flags = [
        _sass_flag(arch_num)
        for arch_num in (80, 86, 89, 90)
        if not known_version or _arch_supported_by_cuda(arch_num, cuda_version)
    ]
    if known_version:
        flags.extend(
            _sass_flag(arch_num)
            for arch_num in (100, 103, 120, 121)  # sm_110/Thor is embedded-only; use the env override for it
            if _arch_supported_by_cuda(arch_num, cuda_version)
        )
    # compute_90 PTX needs CUDA >= 11.8; older or unknown toolchains get
    # compute_80 PTX (torch >= 2.1 ships CUDA >= 11.8, so this branch is
    # exotic-toolchain insurance).
    if known_version and cuda_version >= (11, 8):
        flags.append(_ptx_flag(90))
    else:
        flags.append(_ptx_flag(80))
    return flags


def _arches_from_gencode_flags(flags: list[str]) -> list[str]:
    """Extract SM (SASS) targets like ``["80", "121"]`` from nvcc gencode flags."""
    arches: set[str] = set()
    for flag in flags:
        marker = "code=sm_"
        if marker not in flag:
            continue
        arch = flag.split(marker, 1)[1].split(",", 1)[0].strip()
        if arch.isdigit():
            arches.add(arch)
    return sorted(arches, key=int)


def _ptx_arches_from_gencode_flags(flags: list[str]) -> list[str]:
    """Extract virtual (PTX) targets like ``["90"]`` from nvcc gencode flags."""
    arches: set[str] = set()
    for flag in flags:
        marker = "code=compute_"
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


def _parse_manifest_arches(raw) -> set[str]:
    if not isinstance(raw, list):
        return set()
    parsed = {str(arch).replace(".", "") for arch in raw}
    return {arch for arch in parsed if arch.isdigit()}


def _read_prebuilt_manifest(path: Path) -> tuple[set[str], set[str]] | None:
    """Return ``(sass_arches, ptx_arches)`` from the sidecar manifest.

    ``ptx_arches`` is empty for manifests written before PTX targets were
    recorded — those behave exactly as before.
    """
    manifest = _prebuilt_manifest_path(path)
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read TurboQuant prebuilt manifest %s: %s", manifest, exc)
        return None

    if not isinstance(data.get("arches"), list):
        logger.warning("TurboQuant prebuilt manifest %s has no arches list", manifest)
        return None
    return _parse_manifest_arches(data.get("arches")), _parse_manifest_arches(data.get("ptx"))


# Sentinel distinguishing "manifest not read yet" from "manifest file
# absent/unusable" (None) in _prebuilt_is_compatible's optional parameter.
_MANIFEST_UNREAD = object()


def _prebuilt_is_compatible(path: Path, manifest=_MANIFEST_UNREAD) -> bool:
    """Return whether ``path`` is safe to use on this host.

    A prebuilt extension can import successfully even when it was not built
    for the local GPU. On new architectures (for example GB10 / sm_121),
    silently accepting that path makes performance diagnosis unreliable.
    Bundled builds therefore carry a tiny sidecar manifest with the SM
    targets compiled into the shared object. If a CUDA device is visible, the
    prebuilt must prove it covers every local SM unless the user explicitly
    opts into unverified loading.

    ``manifest`` accepts a pre-read ``_read_prebuilt_manifest`` result so
    callers that also want the manifest (e.g. for logging) parse it once.
    """
    local_arches = set(_detect_local_arches())
    if not local_arches:
        return True

    if _env_flag("TQ_CUDA_ALLOW_UNVERIFIED_PREBUILT"):
        logger.warning(
            "TQ_CUDA_ALLOW_UNVERIFIED_PREBUILT=1: using %s without SM coverage verification for local arches %s",
            path,
            sorted(local_arches, key=int),
        )
        return True

    if manifest is _MANIFEST_UNREAD:
        manifest = _read_prebuilt_manifest(path)
    if manifest is None:
        logger.warning(
            "Skipping unverified TurboQuant prebuilt extension %s on CUDA arches %s; "
            "falling back to local JIT. Set TQ_CUDA_ALLOW_UNVERIFIED_PREBUILT=1 to override.",
            path,
            sorted(local_arches, key=int),
        )
        return False

    prebuilt_arches, ptx_arches = manifest
    missing = local_arches - prebuilt_arches
    # A local arch without exact SASS is still covered when the prebuilt
    # embeds PTX at or below that arch — the driver JIT-compiles PTX forward
    # (e.g. compute_90 PTX runs on sm_100 B200 / sm_103 B300 hardware).
    missing = {arch for arch in missing if not any(int(ptx) <= int(arch) for ptx in ptx_arches)}
    if missing:
        logger.warning(
            "Skipping TurboQuant prebuilt extension %s: manifest arches %s (ptx %s) do not "
            "cover local CUDA arches %s; missing %s. Falling back to local JIT.",
            path,
            sorted(prebuilt_arches, key=int),
            sorted(ptx_arches, key=int),
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
        manifest = _read_prebuilt_manifest(candidate)
        if not _prebuilt_is_compatible(candidate, manifest=manifest):
            continue
        try:
            module = _load_module_from_path(candidate)
            logger.warning(
                "Loaded prebuilt TurboQuant CUDA extension from %s (manifest arches=%s)",
                candidate,
                sorted(manifest[0], key=int) if manifest else [],
            )
            return module
        except Exception as exc:
            logger.warning("Failed to load prebuilt TurboQuant CUDA extension from %s: %s", candidate, exc)
    return None


def _bundle_module(module, arches: list[str], ptx_arches: list[str] | None = None) -> Path:
    """Copy the compiled extension into the package for runtime reuse."""
    PREBUILT_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(module.__file__).resolve()
    target = PREBUILT_DIR / source.name
    if source != target:
        shutil.copy2(source, target)
    manifest = {"arches": arches, "ptx": ptx_arches or []}
    _prebuilt_manifest_path(target).write_text(json.dumps(manifest, indent=2) + "\n")
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
        _bundle_module(
            module,
            _arches_from_gencode_flags(gencode_flags),
            _ptx_arches_from_gencode_flags(gencode_flags),
        )
    return module


if __name__ == "__main__":
    os.environ.setdefault("TQ_CUDA_BUNDLE", "1")
    mod = build()
    print(f"Built successfully: {mod}")
    print(f"Available functions: {[x for x in dir(mod) if not x.startswith('_')]}")

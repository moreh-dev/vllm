# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builds the vendored AITER gfx950 DSA kernels as torch extensions.

A module is its AITER kernel sources (`../third_party/aiter`) plus a binding in
`csrc/`, compiled with AITER's JIT flags against the headers of the installed
AITER package (`aiter_meta/csrc/include`). torch caches the build (a minute or
two on first use) and makes the other ranks wait for the one building it.
"""

import functools
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from types import ModuleType

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_HERE = Path(__file__).resolve().parent
_BINDINGS = _HERE / "csrc"
_AITER = _HERE.parent / "third_party" / "aiter"

# aiter/jit/core.py build_module for HIP >= 6.3, without the gfx1250-only and
# torch-dependent flags.
_AITER_FLAGS = [
    "-O3",
    "-std=c++20",
    "-DLEGACY_HIPBLAS_DIRECT",
    "-DUSE_PROF_API=1",
    "-D__HIP_PLATFORM_HCC__=1",
    "-D__HIP_PLATFORM_AMD__=1",
    "-U__HIP_NO_HALF_CONVERSIONS__",
    "-U__HIP_NO_HALF_OPERATORS__",
    "-D__Float4_e2m1fn_x2",
    "-Wno-unknown-warning-option",
    "-Wno-unused-result",
    "-Wno-switch-bool",
    "-Wno-vla-cxx-extension",
    "-Wno-undefined-func-template",
    "-Wno-macro-redefined",
    "-Wno-missing-template-arg-list-after-template-kw",
    "-fgpu-flush-denormals-to-zero",
    "-fno-offload-uniform-block",
]
# The LLVM options AITER passes; the ones the installed compiler no longer
# knows (e.g. -amdgpu-coerce-illegal-types on ROCm 10) are dropped.
_AITER_LLVM_OPTIONS = [
    "--amdgpu-kernarg-preload-count=32",
    "--lsr-drop-solution=1",
    "-enable-post-misched=0",
    "-amdgpu-early-inline-all=true",
    "-amdgpu-function-calls=false",
    "-amdgpu-coerce-illegal-types=1",
]
# From HIP 7.16 LLVM no longer inlines across differing target features.
_AITER_LLVM_OPTIONS_HIP_7_16 = ["-ignore-tti-inline-compatible=true"]

_FP8_MQA_FLAGS = ["-fno-honor-nans", "-fno-slp-vectorize"]
_FP8_MQA_LLVM_OPTIONS = ["-disable-machine-sink"]

# module -> (kernel sources under third_party/aiter/csrc, module flags, module
# LLVM options), as in AITER's aiter/jit/optCompilerConfig.json.
_MODULES: dict[str, tuple[list[str], list[str], list[str]]] = {
    "kpool_compress": (["kernels/kpool_compress.cu"], [], []),
    "fp8_mqa_logits": (
        ["kernels/fp8_mqa_logits.cu"],
        _FP8_MQA_FLAGS,
        _FP8_MQA_LLVM_OPTIONS,
    ),
    # Device code at -O1: built by the ROCm 10.0 compiler (HIP 7.15) at -O2 or
    # -O3, the candidate scan returns non-top-k pools for about half of the long
    # rows without taking the exact fallback (and runs 2-4x slower).
    # test_glm5next_gfx950_dsa_kernels.py checks the selection is exact.
    "fp8_mqa_topk": (
        ["kernels/fp8_mqa_topk.cu"],
        [*_FP8_MQA_FLAGS, "-Xarch_device", "-O1"],
        _FP8_MQA_LLVM_OPTIONS,
    ),
    "fp8_paged_mqa_logits": (
        ["kernels/fp8_paged_mqa_logits.cu"],
        ["-fno-honor-nans"],
        [],
    ),
    "sparse_mla_fp8": (
        [
            "kernels/mla/sparse_mla_fp8.cu",
            "kernels/mla/fp8/small.cu",
            "kernels/mla/fp8/prefetch.cu",
            "kernels/mla/fp8/rotating.cu",
            "kernels/mla/fp8/streaming.cu",
        ],
        ["-DENABLE_CK=0", "-fno-gpu-flush-denormals-to-zero"],
        [],
    ),
}

_lock = threading.Lock()
_loaded: dict[str, ModuleType] = {}
_failed: dict[str, str] = {}


def _aiter_include_dir() -> Path:
    import aiter_meta

    include = Path(aiter_meta.__file__).resolve().parent / "csrc" / "include"
    if not (include / "aiter_tensor.h").is_file():
        raise RuntimeError(f"the installed AITER has no headers in {include}")
    return include


def _hip_at_least(major: int, minor: int) -> bool:
    version = (torch.version.hip or "0.0").split(".")
    return (int(version[0]), int(version[1])) >= (major, minor)


def _accepts(hipcc: str, source: Path, options: list[str]) -> bool:
    command = [hipcc, "--offload-arch=gfx950", "-c", str(source), "-o", os.devnull]
    for option in options:
        command += ["-mllvm", option]
    return subprocess.run(command, capture_output=True).returncode == 0


@functools.cache
def _llvm_flags(options: tuple[str, ...]) -> tuple[str, ...]:
    """`-mllvm` flags for the `options` the installed compiler accepts."""
    from torch.utils.cpp_extension import ROCM_HOME

    hipcc = os.path.join(ROCM_HOME or "", "bin", "hipcc")
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "probe.hip"
        source.write_text("#include <hip/hip_runtime.h>\n__global__ void probe() {}\n")
        if _accepts(hipcc, source, list(options)):
            kept = list(options)
        else:
            kept = [option for option in options if _accepts(hipcc, source, [option])]
            logger.debug(
                "hipcc rejects the LLVM options %s",
                sorted(set(options) - set(kept)),
            )
    return tuple(flag for option in kept for flag in ("-mllvm", option))


def _sync_tree(src: Path, dst: Path) -> None:
    """Copy the files of `src` that `dst` lacks or holds differently. hipify
    writes next to its inputs, so the build works on a copy; replacing files
    atomically keeps a rank that is already compiling safe."""
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        target = dst / path.relative_to(src)
        data = path.read_bytes()
        if target.is_file() and target.read_bytes() == data:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.{os.getpid()}")
        partial.write_bytes(data)
        os.replace(partial, target)


def _build(module: str) -> ModuleType:
    from torch.utils.cpp_extension import _get_build_directory, load

    kernels, module_flags, module_llvm_options = _MODULES[module]
    name = f"vllm_glm5next_{module}"
    build_dir = Path(_get_build_directory(name, verbose=False))
    src = build_dir / "src"
    _sync_tree(_BINDINGS, src / "bindings")
    _sync_tree(_AITER / "csrc", src / "aiter")
    sources = [str(src / "bindings" / f"{module}_binding.cu")]
    sources += [str(src / "aiter" / kernel) for kernel in kernels]
    llvm_options = _AITER_LLVM_OPTIONS + module_llvm_options
    if _hip_at_least(7, 16):
        llvm_options += _AITER_LLVM_OPTIONS_HIP_7_16
    # A plain -I, not extra_include_paths: hipify would rewrite the installed
    # AITER headers it is pointed at.
    flags = [
        *_AITER_FLAGS,
        *_llvm_flags(tuple(llvm_options)),
        f"-I{_aiter_include_dir()}",
        *module_flags,
    ]

    # vLLM's ROCm image lists every target in PYTORCH_ROCM_ARCH; the kernels
    # are gfx950 code (they trap on any other target).
    saved = os.environ.get("PYTORCH_ROCM_ARCH")
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    try:
        return load(
            name=name,
            sources=sources,
            extra_include_paths=[str(src / "bindings"), str(src / "aiter/include")],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=flags,
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if saved is None:
            del os.environ["PYTORCH_ROCM_ARCH"]
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = saved


def load_module(module: str) -> ModuleType:
    """The built extension for `module`, building it on first use. Raises if
    it does not build (and on every later call in the process)."""
    with _lock:
        if module in _loaded:
            return _loaded[module]
        if module in _failed:
            raise RuntimeError(_failed[module])
        logger.info(
            "Building the HIP %s kernel for gfx950 (cached after the first build)",
            module,
        )
        try:
            ext = _build(module)
        except Exception as err:
            _failed[module] = f"building the HIP {module} kernel failed: {err}"
            raise RuntimeError(_failed[module]) from err
        _loaded[module] = ext
        return ext

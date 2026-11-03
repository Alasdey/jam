"""
The GPU runtime every module shares: CuPy, compiled kernels, loud warnings.

jam runs on one NVIDIA GPU through CuPy. Kernels live in .cu files and are
compiled once per process with the core/cuda headers on the include path
(CuPy caches the binaries on disk, so later processes start fast).
"""

import sys
from functools import cache
from pathlib import Path

CUDA_DIR = Path(__file__).with_name("cuda")


@cache
def cupy():
    """The cupy module, after checking that a CUDA device is there."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "jam runs on the GPU and needs CuPy built for your CUDA driver "
            "(cupy-cuda13x or cupy-cuda12x; nvidia-smi shows the driver's CUDA version)"
        ) from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("jam runs on the GPU and found no CUDA device")
    return cp


@cache
def module(path: str):
    """The compiled RawModule of a .cu file (path relative to the project root, or absolute)."""
    cp = cupy()
    source = Path(path)
    if not source.is_absolute():
        source = Path(__file__).resolve().parent.parent / source
    return cp.RawModule(code=source.read_text(), options=("--std=c++17", f"-I{CUDA_DIR}"))


def kernel(path: str, name: str):
    return module(path).get_function(name)


def launch(fn, n_threads: int, *args, block: int = 128) -> None:
    """Run fn over at least n_threads threads, in blocks of `block`."""
    if n_threads > 0:
        fn(((n_threads + block - 1) // block,), (block,), args)


def warp_launch(fn, n_jobs: int, *args, block: int = 128) -> None:
    """Run fn with one warp (32 threads) per job."""
    launch(fn, 32 * n_jobs, *args, block=block)


def exclusive_cumsum(values):
    """[0, v0, v0 + v1, ...] without the total, as int64, on the array's device."""
    cp = cupy()
    values = cp.asarray(values, dtype=cp.int64)
    out = cp.zeros(len(values), dtype=cp.int64)
    if len(values) > 1:
        cp.cumsum(values[:-1], out=out[1:])
    return out


_RED, _BOLD, _RESET = "\033[31m", "\033[1m", "\033[0m"


def loud_warning(message: str) -> None:
    """
    A banner on stderr, every time: for work that left the GPU (host fallbacks).
    Not warnings.warn, which prints each warning once.
    """
    bar = "!" * 78
    print(f"{_RED}{_BOLD}{bar}\n!! SLOW HOST PATH: {message}\n{bar}{_RESET}", file=sys.stderr, flush=True)

"""
Random numbers for one generation, on the device.

Every draw in jam comes from Philox4x32-10 (core/cuda/philox.cuh) with the
run seed as key and the counter (element, draw, generation, call). Draws are
pure functions of where they are made, not of a stored state, so a run
resumed at generation g draws exactly what the uninterrupted run drew, and
there is no generator state to checkpoint.

Each consumer of randomness in a generation (an array draw here, or a kernel
that draws inside, via kernel_args) takes the next call number. Calls happen
in program order, so a generation's draws are reproducible as long as the
code and config are the same.

Selection steps and creation code take a DeviceRng and draw from it:

    u = rng.uniform(n)            # float64 in [0, 1)
    k = rng.integers(high, n)     # int64 in [0, high), high an int or an array
    p = rng.permutation(n)        # a uniform random permutation of range(n)
"""

import numpy as np

from core.device import cupy, kernel, launch

_SOURCE = "core/cuda/rng.cu"
_MASK = 0xFFFFFFFF


class DeviceRng:
    def __init__(self, seed: int, gen: int):
        if seed < 0 or seed >> 64:
            raise ValueError("seed must fit in 64 unsigned bits")
        if gen < 0 or gen >> 32:
            raise ValueError("gen must fit in 32 unsigned bits")
        self.seed = seed
        self.gen = gen
        self.calls = 0

    def kernel_args(self) -> tuple:
        """(key0, key1, gen, call) for a kernel that draws with rng_init: one call."""
        call = self.calls
        self.calls += 1
        if call >> 32:
            raise RuntimeError("more than 2**32 random calls in one generation")
        return (np.uint32(self.seed & _MASK), np.uint32(self.seed >> 32),
                np.uint32(self.gen), np.uint32(call))

    def uniform(self, n: int):
        cp = cupy()
        out = cp.empty(int(n), dtype=cp.float64)
        args = self.kernel_args()
        launch(kernel(_SOURCE, "rng_uniform_fill"), len(out), out, np.int64(len(out)), *args)
        return out

    def integers(self, high, n: int | None = None):
        """Uniform int64 in [0, high): high an int (then n draws) or a device array (one draw each)."""
        cp = cupy()
        if isinstance(high, (int, np.integer)):
            if n is None:
                raise ValueError("n is required when high is a scalar")
            highs, scalar = cp.full(1, int(high), dtype=cp.int64), 1
        else:
            highs, scalar = cp.asarray(high, dtype=cp.int64), 0
            n = len(highs)
        if len(highs) and (bool((highs < 1).any()) or bool((highs > _MASK).any())):
            raise ValueError("high must be between 1 and 2**32 - 1")
        out = cp.empty(int(n), dtype=cp.int64)
        args = self.kernel_args()
        launch(kernel(_SOURCE, "rng_below_fill"), len(out), out, highs, np.int64(len(out)), np.int32(scalar), *args)
        return out

    def permutation(self, n: int):
        cp = cupy()
        return cp.argsort(self.uniform(n), kind="stable").astype(cp.int64)

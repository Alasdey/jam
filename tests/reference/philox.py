"""NumPy Philox4x32-10, the reference for core/cuda/philox.cuh."""

import numpy as np

MASK = np.uint64(0xFFFFFFFF)


def philox4x32_10(c0, c1, c2, c3, k0, k1):
    c = [np.asarray(v, dtype=np.uint64) & MASK for v in (c0, c1, c2, c3)]
    k0, k1 = np.uint64(k0), np.uint64(k1)
    for r in range(10):
        if r:
            k0 = (k0 + np.uint64(0x9E3779B9)) & MASK
            k1 = (k1 + np.uint64(0xBB67AE85)) & MASK
        p0 = np.uint64(0xD2511F53) * c[0]
        p1 = np.uint64(0xCD9E8D57) * c[2]
        c = [(p1 >> np.uint64(32)) ^ c[1] ^ k0, p1 & MASK, (p0 >> np.uint64(32)) ^ c[3] ^ k1, p0 & MASK]
    return c


def uniform(seed: int, gen: int, call: int, n: int) -> np.ndarray:
    """What DeviceRng(seed, gen).uniform(n) returns as its `call`-th call."""
    i = np.arange(n, dtype=np.uint64)
    out = philox4x32_10(i, 0, gen, call, seed & 0xFFFFFFFF, seed >> 32)
    a = (out[0] >> np.uint64(5)).astype(np.float64)
    b = (out[1] >> np.uint64(6)).astype(np.float64)
    return (a * 67108864.0 + b) / 9007199254740992.0

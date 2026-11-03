"""Iterated elimination of strictly dominated strategies, on the device."""

from typing import Optional

import numpy as np

from core.device import cupy, kernel, launch

_SOURCE = "selection/selection.cu"


def iterated_elimination(payoff, max_iter: Optional[int] = None):
    """
    Surviving row indices (device int64) of a square payoff matrix after
    iterated elimination of strictly dominated rows, comparing over the
    surviving columns only (symmetric IESDS).

    Row i is strictly dominated by j if payoff[j, k] >= payoff[i, k] for all
    active k and payoff[j, k] > payoff[i, k] for at least one. Every round
    removes all currently dominated rows, until none is left (or max_iter).
    """
    cp = cupy()
    A = cp.asarray(payoff, dtype=cp.int32)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("payoff must be a square 2D array")
    n = len(A)
    active = cp.arange(n, dtype=cp.int64)
    for _ in range(n if max_iter is None else max_iter):
        k = len(active)
        if k <= 1:
            break
        MT = cp.ascontiguousarray(A[cp.ix_(active, active)].T)
        dominated = cp.zeros(k, dtype=cp.uint8)
        launch(kernel(_SOURCE, "dominated_rows"), k * k, MT, np.int32(k), dominated)
        if not bool(dominated.any()):
            break
        active = active[dominated == 0]
    return active

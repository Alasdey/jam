"""Lexicase survivor selection over payoff columns (higher is better), on the device."""

import operator

import numpy as np

from core.device import cupy, kernel, launch, warp_launch

_SOURCE = "selection/selection.cu"


def lexicase_selection(payoff, n_accepted: int, rng):
    """Return sorted, unique survivor indices (device int64) selected without replacement.

    Each pick starts with all remaining rows and a fresh random ordering of
    cases (opponent columns). Retain only candidates with the exact best
    payoff on each case, stopping when one remains. Break final ties uniformly.
    All original columns remain available, including already selected opponents.

    Rows with identical payoff vectors are filtered as one group (no case can
    separate them), and a pick stops as soon as one group remains: the same
    candidates, without scanning the remaining cases.
    """
    cp = cupy()
    payoff = cp.asarray(payoff)
    if payoff.ndim != 2:
        raise ValueError("payoff must be a 2D array")
    n_accepted = operator.index(n_accepted)
    if n_accepted < 0:
        raise ValueError("n_accepted must be non-negative")
    payoff = payoff.astype(cp.int32)
    n_rows, n_cases = payoff.shape
    if n_rows <= n_accepted:
        return cp.arange(n_rows, dtype=cp.int64)
    if not n_accepted:
        return cp.zeros(0, dtype=cp.int64)
    order = cp.argsort(rng.uniform(n_accepted * n_cases).reshape(n_accepted, n_cases), axis=1)
    tie = rng.uniform(n_accepted)
    return _lexicase(payoff, n_accepted, order.astype(cp.int32), tie)


def _lexicase(payoff, n_accepted: int, order, tie):
    """The selection with explicit case orders (n_accepted, n_cases) and tie draws in [0, 1)."""
    cp = cupy()
    n_rows, n_cases = payoff.shape
    group_of, rep = identical_row_groups(payoff)
    left = cp.bincount(group_of, minlength=len(rep)).astype(cp.int32)
    selected = cp.empty(n_accepted, dtype=cp.int64)
    kernel(_SOURCE, "lexicase")(
        (1,), (1024,),
        (cp.ascontiguousarray(payoff.T), np.int32(n_rows), np.int32(n_cases), np.int32(len(rep)), rep, group_of,
         left, cp.zeros(len(rep), dtype=cp.uint8), cp.zeros(n_rows, dtype=cp.uint8),
         cp.ascontiguousarray(order, dtype=cp.int32), cp.ascontiguousarray(tie, dtype=cp.float64),
         np.int32(n_accepted), selected),
    )
    return cp.sort(selected)


def identical_row_groups(payoff):
    """(group of each row, one representative row per group): rows are grouped iff equal."""
    cp = cupy()
    payoff = cp.ascontiguousarray(payoff, dtype=cp.int32)
    n_rows, n_cols = payoff.shape
    hashes = cp.empty(n_rows, dtype=cp.uint64)
    launch(kernel(_SOURCE, "row_hash"), n_rows, payoff, np.int32(n_rows), np.int32(n_cols), hashes)
    index = cp.arange(n_rows, dtype=cp.int64)
    order = cp.lexsort(cp.stack([index, hashes.astype(cp.int64)]))
    h = hashes[order]
    start = cp.ones(n_rows, dtype=bool)
    start[1:] = h[1:] != h[:-1]
    run = cp.cumsum(start) - 1
    leader = order[cp.flatnonzero(start)][run]  # smallest row of each run of equal hashes
    same = cp.empty(n_rows, dtype=cp.uint8)
    warp_launch(kernel(_SOURCE, "rows_equal"), n_rows, payoff, np.int32(n_cols), leader, order, same,
                np.int32(n_rows))
    # A hash collision (equal hash, different row) leaves the row its own group.
    label = cp.empty(n_rows, dtype=cp.int64)
    label[order] = cp.where(same.astype(bool), leader, order)
    rep, group_of = cp.unique(label, return_inverse=True)
    return group_of.astype(cp.int64), rep.astype(cp.int64)

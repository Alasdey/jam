"""Select programs whose total payoffs lie furthest from their nearest neighbors."""

import operator

from core.device import cupy
from selection.knn_novelty import rank_descending

_BLOCK = 1024


def knn_score_selection(payoff, n_accepted: int, k: int, rng):
    """Keep the largest mean distances to k nearest other scalar scores.

    A program's sole coordinate is its total payoff against the rest of the
    population, excluding self-play. Distance is the absolute score difference.
    Exclude the program itself as a neighbor, but include other programs with
    equal scores. Clamp k to the available neighbors and compute all distances
    against the full input population once, before selecting survivors.
    Return sorted, unique indices; break ties randomly.
    """
    cp = cupy()
    payoff = cp.asarray(payoff)
    if payoff.ndim != 2 or payoff.shape[0] != payoff.shape[1]:
        raise ValueError("payoff must be a square 2D array")
    n_accepted = operator.index(n_accepted)
    k = operator.index(k)
    if n_accepted < 0:
        raise ValueError("n_accepted must be non-negative")
    if k < 1:
        raise ValueError("k must be positive")
    n = len(payoff)
    if n_accepted >= n:
        return cp.arange(n, dtype=cp.int64)
    if n_accepted == 0:
        return cp.zeros(0, dtype=cp.int64)
    scores = _knn_score_distances(payoff, k)
    return cp.sort(rank_descending(scores, rng)[:n_accepted])


def _knn_score_distances(payoff, k: int):
    """Mean distance of each total score to its k nearest other scores (n >= 2)."""
    cp = cupy()
    payoff = cp.asarray(payoff, dtype=cp.float64)
    n = len(payoff)
    k = min(k, n - 1)
    coordinates = payoff.sum(axis=1) - payoff.diagonal()
    out = cp.empty(n, dtype=cp.float64)
    for start in range(0, n, _BLOCK):
        stop = min(start + _BLOCK, n)
        distances = cp.abs(coordinates[start:stop, None] - coordinates[None, :])
        distances[cp.arange(stop - start), cp.arange(start, stop)] = cp.inf
        out[start:stop] = cp.partition(distances, k - 1, axis=1)[:, :k].mean(axis=1)
    return out

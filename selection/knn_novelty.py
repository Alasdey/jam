"""Select isolated reward profiles, without rewarding wins or total payoff."""

import operator

from core.device import cupy

_BLOCK = 1024  # rows of distances computed at once, bounding memory to _BLOCK x N


def knn_novelty_selection(payoff, n_accepted: int, k: int, rng):
    """Keep the largest mean distances to k nearest other payoff rows.

    Each row is a position whose coordinates are rewards against the same
    opponent columns (including the self-play column). Exclude the individual
    itself as a neighbor, but retain other individuals with identical rows.
    Clamp k to the available neighbors. Score once against the full input
    population, then select without replacement; break ties randomly.
    """
    cp = cupy()
    payoff = cp.asarray(payoff)
    if payoff.ndim != 2:
        raise ValueError("payoff must be a 2D array")
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
    scores = _knn_novelty_distances(payoff, k)
    return cp.sort(rank_descending(scores, rng)[:n_accepted])


def rank_descending(scores, rng):
    """Indices by decreasing score, ties in uniformly random order."""
    cp = cupy()
    shuffled = rng.permutation(len(scores))
    return shuffled[cp.argsort(-scores[shuffled], kind="stable")]


def _knn_novelty_distances(payoff, k: int):
    """Mean Euclidean distance of each row to its k nearest other rows (n >= 2)."""
    cp = cupy()
    payoff = cp.asarray(payoff, dtype=cp.float64)
    n = len(payoff)
    k = min(k, n - 1)
    norms = (payoff * payoff).sum(axis=1)
    scores = cp.empty(n, dtype=cp.float64)
    for start in range(0, n, _BLOCK):
        stop = min(start + _BLOCK, n)
        distances = payoff[start:stop] @ payoff.T
        distances *= -2
        distances += norms[start:stop, None]
        distances += norms[None, :]
        cp.maximum(distances, 0, out=distances)
        distances[cp.arange(stop - start), cp.arange(start, stop)] = cp.inf
        nearest = cp.partition(distances, k - 1, axis=1)[:, :k]
        scores[start:stop] = cp.sqrt(nearest).mean(axis=1)
    return scores

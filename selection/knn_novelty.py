"""Select isolated reward profiles, without rewarding wins or total payoff."""

import operator

import numpy as np


def knn_novelty_selection(payoff: np.ndarray, n_accepted: int, k: int) -> np.ndarray:
    """Keep the largest mean distances to k nearest other payoff rows.

    Each row is a position whose coordinates are rewards against the same
    opponent columns (including the self-play column). Exclude the individual
    itself as a neighbor, but retain other individuals with identical rows.
    Clamp k to the available neighbors. Score once against the full input
    population, then select without replacement; break ties with NumPy's RNG.
    Compute distances in blocks to avoid an N x N x N difference tensor.
    """
    payoff = np.asarray(payoff, dtype=np.float64)
    if payoff.ndim != 2:
        raise ValueError("payoff must be a 2D array")
    if not np.isfinite(payoff).all():
        raise ValueError("payoff must contain only finite values")
    n_accepted = operator.index(n_accepted)
    k = operator.index(k)
    if n_accepted < 0:
        raise ValueError("n_accepted must be non-negative")
    if k < 1:
        raise ValueError("k must be positive")
    n = len(payoff)
    if n_accepted >= n:
        return np.arange(n)
    if n_accepted == 0:
        return np.empty(0, dtype=int)
    k = min(k, n - 1)
    norms = np.einsum("ij,ij->i", payoff, payoff)
    scores = np.empty(n)
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        distances = payoff[start:stop] @ payoff.T
        distances *= -2
        distances += norms[start:stop, None]
        distances += norms[None, :]
        np.maximum(distances, 0, out=distances)
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        distances.partition(k - 1, axis=1)
        scores[start:stop] = np.sqrt(distances[:, :k]).mean(axis=1)
    shuffled = np.random.permutation(n)
    ranked = shuffled[np.argsort(-scores[shuffled], kind="stable")]
    return np.sort(ranked[:n_accepted])

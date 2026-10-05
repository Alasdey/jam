"""Select programs whose total payoffs lie closest to their neighbors' totals."""

import operator

import numpy as np


def knn_score_selection(payoff: np.ndarray, n_accepted: int, k: int) -> np.ndarray:
    """Keep the smallest mean distances to k nearest other scalar scores.

    A program's sole coordinate is its total payoff against the rest of the
    population, excluding self-play. Distance is the absolute score difference.
    Exclude the program itself as a neighbor, but include other programs with
    equal scores. Clamp k to the available neighbors and compute all distances
    against the full input population once, before selecting survivors.
    Return sorted, unique indices; break ties using NumPy's global RNG.
    """
    payoff = np.asarray(payoff, dtype=np.float64)
    if payoff.ndim != 2 or payoff.shape[0] != payoff.shape[1]:
        raise ValueError("payoff must be a square 2D array")
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
    coordinates = payoff.sum(axis=1) - payoff.diagonal()
    mean_distances = np.empty(n)
    # Bound extra distance storage to 256 x N instead of another N x N array.
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        distances = np.abs(coordinates[start:stop, None] - coordinates[None, :])
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        distances.partition(k - 1, axis=1)
        mean_distances[start:stop] = distances[:, :k].mean(axis=1)

    shuffled = np.random.permutation(n)
    ranked = shuffled[np.argsort(mean_distances[shuffled], kind="stable")]
    return np.sort(ranked[:n_accepted])

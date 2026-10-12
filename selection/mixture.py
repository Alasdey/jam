"""Mix the best payoffs, scalar novelty among good programs, and row novelty."""

import operator

import numpy as np

from selection.knn_novelty import _knn_novelty_distances
from selection.knn_score import _knn_score_distances


def mixture_selection(
    payoff: np.ndarray,
    n_best: int,
    n_good: int,
    n_good_novel: int,
    n_novelty: int,
    k: int = 10,
) -> np.ndarray:
    """Return sorted, unique survivors from three sequential, disjoint groups.

    Keep n_best by payoff row-sum. Among the next n_good by row-sum, keep
    n_good_novel by scalar kNN distance (the existing knn_score criterion).
    Finally keep n_novelty by payoff-row kNN distance (knn_novelty) among all
    unselected programs, including those passed over in the good pool.

    Both kNN criteria use the full input population as neighbors, with scores
    computed once and the same k, clamped to the available neighbors. Scalar
    kNN excludes self-play; payoff-row novelty includes all original columns.
    Break ties using NumPy's global RNG. Oversized quotas keep everyone.
    """
    payoff = np.asarray(payoff, dtype=np.float64)
    if payoff.ndim != 2 or payoff.shape[0] != payoff.shape[1]:
        raise ValueError("payoff must be a square 2D array")
    if not np.isfinite(payoff).all():
        raise ValueError("payoff must contain only finite values")
    n_best, n_good, n_good_novel, n_novelty, k = map(
        operator.index, (n_best, n_good, n_good_novel, n_novelty, k)
    )
    if min(n_best, n_good, n_good_novel, n_novelty) < 0:
        raise ValueError("selection counts must be non-negative")
    if n_good_novel > n_good:
        raise ValueError("n_good_novel must not exceed n_good")
    if k < 1:
        raise ValueError("k must be positive")

    n = len(payoff)
    total = n_best + n_good_novel + n_novelty
    if total >= n:
        return np.arange(n)
    if total == 0:
        return np.empty(0, dtype=int)

    # One random ordering provides reproducible tie-breaking at every stage.
    shuffled = np.random.permutation(n)
    scores = payoff.sum(axis=1)
    ranked = shuffled[np.argsort(-scores[shuffled], kind="stable")]
    best = ranked[:n_best]
    good = ranked[n_best:n_best + n_good]
    if n_good_novel:
        scalar_novelty = _knn_score_distances(payoff, k)
        # Use the random ordering for novelty ties, independently of payoff rank.
        good = shuffled[np.isin(shuffled, good)]
        good = good[np.argsort(-scalar_novelty[good], kind="stable")]
    good_novel = good[:n_good_novel]

    keep = np.zeros(n, dtype=bool)
    keep[best] = True
    keep[good_novel] = True
    if n_novelty:
        novelty = _knn_novelty_distances(payoff, k)
        remaining = shuffled[~keep[shuffled]]
        remaining = remaining[np.argsort(-novelty[remaining], kind="stable")]
        keep[remaining[:n_novelty]] = True
    return np.flatnonzero(keep)

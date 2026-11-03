"""Select the best scores, the worst scores, and random remaining programs."""

import operator

from core.device import cupy
from selection.knn_novelty import rank_descending


def best_worst_random_selection(payoff, n_best: int, n_worst: int, n_rand: int, rng):
    """Keep disjoint groups by total payoff against the full input population.

    Scores are payoff row-sums, as in cap_top. Pick the best first, then the
    worst among those left, then sample uniformly without replacement from
    the remainder. Break score ties randomly. Return sorted, unique indices;
    oversized quotas keep everyone.
    """
    cp = cupy()
    payoff = cp.asarray(payoff)
    if payoff.ndim != 2 or payoff.shape[0] != payoff.shape[1]:
        raise ValueError("payoff must be a square 2D array")
    n_best, n_worst, n_rand = map(operator.index, (n_best, n_worst, n_rand))
    if min(n_best, n_worst, n_rand) < 0:
        raise ValueError("n_best, n_worst, and n_rand must be non-negative")
    n = len(payoff)
    if n_best + n_worst + n_rand >= n:
        return cp.arange(n, dtype=cp.int64)
    if n_best + n_worst + n_rand == 0:
        return cp.zeros(0, dtype=cp.int64)

    ranked = rank_descending(payoff.sum(axis=1, dtype=cp.int64), rng)
    best = ranked[:n_best]
    split = n - n_worst  # avoids a [-0:] slice, which would be everyone
    worst = ranked[split:]
    remaining = ranked[n_best:split]
    randoms = remaining[rng.permutation(len(remaining))[:n_rand]]
    return cp.sort(cp.concatenate([best, worst, randoms]))

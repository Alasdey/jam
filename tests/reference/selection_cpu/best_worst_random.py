"""Select the best scores, the worst scores, and random remaining programs."""

import operator

import numpy as np


def best_worst_random_selection(
    payoff: np.ndarray, n_best: int, n_worst: int, n_rand: int
) -> np.ndarray:
    """Keep disjoint groups by total payoff against the full input population.

    Scores are payoff row-sums, as in cap_top. Pick the best first, then the
    worst among those left, then sample uniformly without replacement from
    the remainder. Break score ties using NumPy's global RNG for reproducible
    seeded runs. Return sorted, unique indices; oversized quotas keep everyone.
    """
    payoff = np.asarray(payoff, dtype=np.float64)
    if payoff.ndim != 2 or payoff.shape[0] != payoff.shape[1]:
        raise ValueError("payoff must be a square 2D array")
    if not np.isfinite(payoff).all():
        raise ValueError("payoff must contain only finite values")
    n_best, n_worst, n_rand = map(operator.index, (n_best, n_worst, n_rand))
    if min(n_best, n_worst, n_rand) < 0:
        raise ValueError("n_best, n_worst, and n_rand must be non-negative")

    n = len(payoff)
    if n_best + n_worst + n_rand >= n:
        return np.arange(n)
    if n_best + n_worst + n_rand == 0:
        return np.empty(0, dtype=int)

    scores = payoff.sum(axis=1)
    shuffled = np.random.permutation(n)
    ranked = shuffled[np.argsort(-scores[shuffled], kind="stable")]
    best = ranked[:n_best]
    # Avoid a [-0:] slice, which would select the entire population.
    split = n - n_worst
    worst = ranked[split:]
    remaining = ranked[n_best:split]
    randoms = np.random.choice(remaining, size=n_rand, replace=False)
    return np.sort(np.concatenate([best, worst, randoms]))

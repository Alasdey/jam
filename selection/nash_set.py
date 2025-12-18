# utils/nash_subset.py

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import nashpy as nash


def compute_nash_equilibrium(
    payoff: np.ndarray,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, List[int], float]:
    """
    Compute a Nash equilibrium for a zero-sum game given the row player's
    payoff matrix and return the row strategy, its support, and the game value.

    We assume:
        - payoff[i, j] is the payoff for the row player
        - the column player gets -payoff[i, j] (zero-sum)

    Args:
        payoff:
            2D array of shape (n_rows, n_cols) with payoffs to the row player.
        tol:
            Threshold below which probabilities are treated as zero when
            extracting the support.

    Returns:
        row_strategy:
            1D numpy array of length n_rows with the row player's mixed strategy.
        support:
            List of row indices i with row_strategy[i] > tol (the Nash subset).
        value:
            Scalar game value at equilibrium.
    """
    payoff = np.asarray(payoff, dtype=float)

    A = payoff
    B = -A
    game = nash.Game(A, B)

    # Take the first Nash equilibrium found by support enumeration
    the_point = game.support_enumeration()
    row_strategy, col_strategy = next(the_point)

    row_strategy = np.asarray(row_strategy, dtype=float)
    col_strategy = np.asarray(col_strategy, dtype=float)

    support = [i for i, p in enumerate(row_strategy) if p > tol]

    # Game value for the row player
    value = float(row_strategy @ A @ col_strategy)

    return row_strategy, support, value, the_point


def compute_nash_subset(
    payoff: np.ndarray,
    tol: float = 1e-8,
) -> List[int]:
    """
    Return the set of row indices that get positive probability in at least
    one Nash equilibrium (union of supports over all equilibria).
    """
    payoff = np.asarray(payoff, dtype=float)
    A = payoff
    B = -A
    game = nash.Game(A, B)

    n_rows = A.shape[0]
    in_some_equilibrium = np.zeros(n_rows, dtype=bool)

    for row_strategy, _ in game.support_enumeration():
        row_strategy = np.asarray(row_strategy, dtype=float)
        in_some_equilibrium |= row_strategy > tol

    return [i for i, flag in enumerate(in_some_equilibrium) if flag]

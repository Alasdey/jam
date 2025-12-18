import numpy as np
from typing import Iterable, Optional


def iterated_elimination_strictly_dominated_rows(
    payoff: np.ndarray,
    atol: float = 0.0,
) -> np.ndarray:
    """
    Perform iterative elimination of strictly dominated row strategies.

    A row i is strictly dominated by j if:
        payoff[j, k] >= payoff[i, k]  for all k
    and
        payoff[j, k] >  payoff[i, k]  for at least one k.

    This is applied iteratively: once a row is removed, it no longer
    participates in domination checks. We continue until no further
    eliminations are possible.

    Args:
        payoff:
            2D numpy array of shape (n_rows, n_cols),
            giving payoffs for each row strategy.
        atol:
            Optional tolerance for floating-point comparisons. If > 0,
            comparisons are done as:
                ge: payoff[j, :] >= payoff[i, :] - atol
                gt: payoff[j, :] >  payoff[i, :] + atol

    Returns:
        A 1D numpy array containing the indices (w.r.t. the original payoff
        matrix) of the surviving row strategies.
    """
    A = np.asarray(payoff, dtype=float)
    n_rows, n_cols = A.shape

    # Track which original rows are still active
    active = np.arange(n_rows)

    changed = True
    while changed:
        changed = False
        num_active = len(active)

        # Try to find any dominated row among the active ones
        for idx_pos_i in range(num_active):
            i = active[idx_pos_i]
            row_i = A[i]

            dominated = False
            for idx_pos_j in range(num_active):
                j = active[idx_pos_j]
                if i == j:
                    continue

                row_j = A[j]

                # j >= i componentwise (within atol)
                ge = np.all(row_j >= row_i - atol)
                # and strictly better in at least one component (beyond atol)
                gt = np.any(row_j > row_i + atol)

                if ge and gt:
                    # i is strictly dominated by j
                    dominated = True
                    break

            if dominated:
                # Remove row i from active set and restart search
                active = np.delete(active, idx_pos_i)
                changed = True
                break

    return active



def iterated_elimination_strictly_dominated_rows_fast(
    payoff: np.ndarray,
    atol: float = 0.0,
    max_iter: Optional[int] = None,
) -> np.ndarray:
    """
    Faster iterative elimination of strictly dominated row strategies.

    A row i is strictly dominated by j if:
        payoff[j, k] >= payoff[i, k]  for all k
    and
        payoff[j, k] >  payoff[i, k]  for at least one k.

    We apply this iteratively until no more rows can be removed
    (or until max_iter is reached, if provided).

    Args:
        payoff:
            2D numpy array (n_rows, n_cols).
        atol:
            Tolerance for float comparisons.
        max_iter:
            Optional safety cap on number of outer iterations
            (usually not needed; elimination converges quickly).

    Returns:
        1D numpy array of surviving row indices w.r.t. original payoff.
    """
    A = np.asarray(payoff, dtype=float)
    n_rows, n_cols = A.shape

    # Start with all rows active (indices in original matrix)
    active = np.arange(n_rows)

    # Optional safety cap
    if max_iter is None:
        max_iter = n_rows  # can't eliminate more than n_rows anyway

    for _ in range(max_iter):
        k = len(active)
        if k <= 1:
            break  # nothing left to dominate

        # Submatrix of active rows: shape (k, n_cols)
        M = A[active]  # view/copy depending on strides, but cheap enough

        # Compare all row pairs via broadcasting:
        # M[:, None, :] shape (k,1,n_cols)
        # M[None, :, :] shape (1,k,n_cols)
        # ge[j, i] = True iff row j >= row i componentwise (within atol)
        ge = M[:, None, :] >= (M[None, :, :] - atol)

        # gt[j, i] = True iff row j > row i in at least one column (beyond atol)
        gt = M[:, None, :] > (M[None, :, :] + atol)

        # ge_all[j, i]: componentwise >= in every column
        ge_all = np.all(ge, axis=2)
        # gt_any[j, i]: strictly better in at least one column
        gt_any = np.any(gt, axis=2)

        # dominance[j, i] = True if j strictly dominates i
        dominance = ge_all & gt_any

        # a row can't dominate itself
        np.fill_diagonal(dominance, False)

        # Row i is dominated if exists some j with dominance[j, i] = True
        dominated = np.any(dominance, axis=0)  # shape (k,)

        if not np.any(dominated):
            # Fixed point reached: no more eliminations
            break

        # Keep only non-dominated rows (in terms of current active set)
        active = active[~dominated]

    return active

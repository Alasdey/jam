"""Lexicase survivor selection over payoff columns (higher is better)."""

import operator

import numpy as np


# A performance cutoff only: avoid large score-tier tables for many-valued cases.
_MAX_BITSET_LEVELS = 8


def lexicase_selection(payoff: np.ndarray, n_accepted: int) -> np.ndarray:
    """Return sorted, unique survivor indices selected without replacement.

    Each pick starts with all remaining rows and a fresh random ordering of
    cases (opponent columns). Retain only candidates with the exact best
    payoff on each case, stopping when one remains. Break final ties uniformly.
    All original columns remain available, including already selected opponents.
    Uses NumPy's global RNG, preserving the original algorithm's random draws.

    Filter bitsets of identical payoff-row groups when every case has at most
    eight distinct scores, choosing ties uniformly among individuals. Fall
    back to the original array filtering for cases with more distinct scores.
    """
    payoff = np.asarray(payoff)
    if payoff.ndim != 2:
        raise ValueError('payoff must be a 2D array')
    n_accepted = operator.index(n_accepted)
    if n_accepted < 0:
        raise ValueError('n_accepted must be non-negative')
    if not np.isfinite(payoff).all():
        raise ValueError('payoff must contain only finite values')
    n_rows, n_cases = payoff.shape
    if n_rows <= n_accepted:
        return np.arange(n_rows)
    if not n_accepted:
        return np.empty(0, dtype=int)
    contiguous = np.ascontiguousarray(payoff)
    if n_cases:
        # Equal bytes imply equal finite numeric payoffs. Signed zeros may
        # create separate groups, which are combined in the final tie below.
        row_dtype = np.dtype((np.void, contiguous.dtype.itemsize * n_cases))
        keys = contiguous.view(row_dtype).ravel()
        _, representatives, group_ids = np.unique(
            keys, return_index=True, return_inverse=True
        )
    else:
        representatives = np.array([0])
        group_ids = np.zeros(n_rows, dtype=int)
    n_groups = len(representatives)
    members = [np.flatnonzero(group_ids == group) for group in range(n_groups)]
    case_masks = []
    for scores in payoff[representatives].T:
        levels = np.unique(scores)
        if len(levels) > _MAX_BITSET_LEVELS:
            # Bitsets target discrete payoffs; preserve the ordinary path for
            # cases with many distinct scores. No RNG has been consumed yet.
            return _lexicase_array(payoff, n_accepted)
        # The lowest score needs no mask: if no higher tier intersects the
        # candidate set, all candidates necessarily have that lowest score.
        masks = [
            int.from_bytes(
                np.packbits(scores == score, bitorder="little").tobytes(),
                "little",
            )
            for score in levels[:0:-1]
        ]
        case_masks.append(masks)
    active = (1 << n_groups) - 1
    selected = []
    for _ in range(n_accepted):
        candidates = active
        # Retain the full permutation even when only one group remains, so
        # subsequent evolution consumes the same RNG stream as before.
        for case in np.random.permutation(n_cases):
            if candidates & (candidates - 1) == 0:
                break
            for mask in case_masks[case]:
                survivors = candidates & mask
                if survivors:
                    candidates = survivors
                    break
        if candidates & (candidates - 1) == 0:
            group = candidates.bit_length() - 1
            rows = members[group]
        else:
            # Relevant only for byte-distinct but numerically equal rows,
            # such as signed floating-point zeros.
            groups = []
            while candidates:
                bit = candidates & -candidates
                groups.append(bit.bit_length() - 1)
                candidates ^= bit
            rows = np.sort(np.concatenate([members[g] for g in groups]))
        winner = np.random.choice(rows)
        group = int(group_ids[winner])
        selected.append(winner)
        members[group] = members[group][members[group] != winner]
        if not len(members[group]):
            active &= ~(1 << group)
    return np.sort(np.asarray(selected, dtype=int))


def _lexicase_array(payoff: np.ndarray, n_accepted: int) -> np.ndarray:
    """Original array filtering, for already validated inputs."""
    n_rows, n_cases = payoff.shape
    remaining = np.arange(n_rows)

    selected = []
    for _ in range(n_accepted):
        candidates = remaining
        for case in np.random.permutation(n_cases):
            if len(candidates) == 1:
                break
            scores = payoff[candidates, case]
            candidates = candidates[scores == scores.max()]
        winner = np.random.choice(candidates)
        selected.append(winner)
        remaining = remaining[remaining != winner]

    return np.sort(np.asarray(selected, dtype=int))

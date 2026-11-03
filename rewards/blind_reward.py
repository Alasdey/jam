import numpy as np

from core.matchups import Matchups


def game(m: Matchups, x, y):
    """1 where x run on its own output against y equals y run on x: run(x, run(x, y)) == run(y, x)."""
    return m.run(x, m.run(x, y)).equals(m.run(y, x)).astype(np.int64)


def reward(m: Matchups) -> np.ndarray:
    """A's game against B, 0 or 1: not zero-sum."""
    return game(m, m.a, m.b)

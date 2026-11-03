import numpy as np

from core.matchups import Matchups

# A fixed balanced input (two nested nodes and a leaf): Treemo tapes hold only 0 and 1.
STAPLE = [1, 1, 0, 0, 1, 0]


def reward(m: Matchups) -> np.ndarray:
    """
    Both programs run on the same fixed input; the longer output wins: +1, -1 or 0.
    """
    staple = m.const(STAPLE)
    return np.sign(m.run(m.a, staple).length() - m.run(m.b, staple).length())

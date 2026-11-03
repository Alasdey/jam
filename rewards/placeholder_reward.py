
import numpy as np

from core.matchups import Matchups

STAPLE = [0, 1, 2, 3, 4, 5]


def reward(m: Matchups) -> np.ndarray:
    """
    Reward obtained by playing both side of the subleq game with deterministic reward -1, 0 or 1
    The more the better for A 
    """
    staple = m.const(STAPLE)
    return np.sign(m.run(m.a, staple).length() - m.run(m.b, staple).length())

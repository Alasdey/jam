# rewards/quine_pressure.py

import numpy as np

from core.matchups import Matchups


def reward(m: Matchups) -> np.ndarray:
    """
    Quine Pressure reward.

    A wins  (+1) if A imprints itself on B more than B imprints itself on A.
    Draw    ( 0) if imprint strengths are equal.
    A loses (-1) otherwise.

    A's imprint is the normalised LCS similarity of A's output, when run on B,
    to A itself: how much of A appears in it. Both sides of the interaction are
    evaluated, making this fully symmetric and zero-sum:
    reward(A, B) = -reward(B, A).
    """
    imprint_a = m.run(m.a, m.b).similarity(m.a)  # A runs on B → does the output look like A?
    imprint_b = m.run(m.b, m.a).similarity(m.b)  # B runs on A → does the output look like B?
    return np.sign(imprint_a - imprint_b)

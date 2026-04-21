# rewards/quine_pressure.py

from typing import List, Union

Program = Union[List[int], str]


def _lcs_length(a: List[int], b: List[int]) -> int:
    """
    Standard DP longest common subsequence length.
    O(|a| * |b|) time and O(min(|a|,|b|)) space.
    """
    if not a or not b:
        return 0

    # Keep the shorter sequence in the inner loop
    if len(a) < len(b):
        a, b = b, a

    n = len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for x in a:
        for j, y in enumerate(b):
            curr[j + 1] = prev[j] + 1 if x == y else max(curr[j], prev[j + 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def _similarity(output: Program, reference: Program) -> float:
    """
    Normalised LCS similarity: how much of `reference` appears in `output`.
    Returns 0.0 if either is empty.
    """
    if not output or not reference:
        return 0.0

    # Treemo: operate on character lists
    if isinstance(output, str):
        output = list(output)
        reference = list(reference)

    lcs = _lcs_length(output, reference)
    return lcs / len(reference)


def reward(interpreter, code_a: Program, code_b: Program) -> int:
    """
    Quine Pressure reward.

    A wins  (+1) if A imprints itself on B more than B imprints itself on A.
    Draw    ( 0) if imprint strengths are equal.
    A loses (-1) otherwise.

    Both sides of the interaction are evaluated, making this fully symmetric
    and zero-sum: reward(A, B) = -reward(B, A).

    Args:
        interpreter : any interpreter with .run(code, input) -> (output, mem)
        code_a      : program A (List[int] or str)
        code_b      : program B (List[int] or str)

    Returns:
        +1, 0, or -1
    """
    # A runs on B → does the output look like A?
    out_ab, _ = interpreter.run(code_a, code_b)
    # B runs on A → does the output look like B?
    out_ba, _ = interpreter.run(code_b, code_a)

    imprint_a = _similarity(out_ab, code_a)  # A's self-similarity in output
    imprint_b = _similarity(out_ba, code_b)  # B's self-similarity in output

    score = imprint_a - imprint_b

    if score > 0:
        return 1
    elif score < 0:
        return -1
    return 0
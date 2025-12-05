
from typing import List


def reward(interpreter, code_a: List[int], code_b: List[int]) -> int:
    """
    Reward obtained by playing both side of the subleq game with deterministic reward -1, 0 or 1
    The more the better for A 
    """
    staple = [0, 1, 2, 3, 4, 5]
    out_a = interpreter.run(code_a, staple)
    out_b = interpreter.run(code_b, staple)
    
    if len(out_a) == len(out_b):
        return 0
    elif len(out_a) > len(out_b):
        return 1
    return 0
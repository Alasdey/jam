
from typing import List

from interpreters.subleq.subleq import subleq


def reward(interpreter, code_a: List[int], code_b: List[int], max_output_length: int, max_iterations: int) -> int:
    """
    Reward obtained by playing both side of the subleq game with deterministic reward -1, 0 or 1
    The more the better for A 
    """
    staple = [0, 1, 2, 3, 4, 5]
    out_a = interpreter(code_a, staple, max_output_length, max_iterations)
    out_b = interpreter(code_b, staple, max_output_length, max_iterations)
    
    if len(out_a) == len(out_b):
        return 0
    elif len(out_a) > len(out_b):
        return 1
    return -1
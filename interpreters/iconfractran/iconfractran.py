# This is an unfinished (maybe derelict) piece of code

from typing import List, Iterator, Tuple

def code_rules(code: List[int]) -> Iterator[Tuple[int, int]]:
    # Pair up (a,b) from code[1:]
    pairs = code[1:]
    for i in range(0, len(pairs) - 1, 2):
        yield pairs[i], pairs[i + 1]

def check(code: List[int], mem: List[int]) -> bool:
    # Apply first applicable rule to first applicable mem cell
    for a, b in code_rules(code):
        for i in range(len(mem)):
            if a == 0:
                continue  # avoid ZeroDivisionError / undefined rule
            if mem[i] % a == 0:  # divisible
                mem[i] = (mem[i] // a) * b
                return True
    return False

def step(code: List[int], mem: List[int]) -> Tuple[List[int], List[int]]:
    # If code[0] is meant to be a special register, try it first
    # We check it as a one-item memory, then write it back if it changed.
    if len(code) > 0:
        reg = [code[0]]
        if check(code=code, mem=reg):
            code[0] = reg[0]
            return code, mem

    # Otherwise (or next), operate on normal memory
    check(code=code, mem=mem)
    return code, mem

def ift(program: List[int], inp: List[int], max_step: int = 2000) -> Tuple[List[int], List[int]]:
    code = program.copy()
    res = inp.copy()
    for _ in range(max_step):
        code, res = step(code=code, mem=res)
    return res, code

class IconfractranInterpreter:
    def __init__(self, max_step: int = 1000):
        self.max_step = max_step

    def run(self, program: List[int], inp: List[int]):
        return ift(program=program, inp=inp, max_step=self.max_step)

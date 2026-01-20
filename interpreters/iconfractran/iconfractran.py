
from typing import List


def code_rules(code: List[int]):
    """
    """
    for i in range(len(code[1:])/2):
        yield code[i*2+1: i*2+2]


def check(code: List[int], mem: List[int]):
    """
    """
    for rule in code_rules(code):
        for i in range(len(mem)):
            if mem[i] % rule[0]:
                mem[i] = int(mem[i] / rule[0] * rule[1])
                return True
    return False


def step(code: List[int], mem: List[int]):
    """
    """
    if check(code=code, mem=code[:1]):
        return code, mem
    if check(code=code, mem=mem):
        return code, mem
    return code, mem


def ift(program: List[int], intput: List[int], max_step: int=2000):
    """
    """
    code = program.copy()
    res = input.copy()
    for _ in range(max_step):
        code, res = step(code=code, mem=res)
    return (res, code)


def IconfractranInterpreter():
    """
    """
    def __init__(
        self,
        max_step=1000,
    ):
        """
        """
        self.max_step = max_step
    
    def run(self, program: List[int], intput: List[int]):
        """
        """
        return ift(program=program, intput=input, max_step=self.max_step)


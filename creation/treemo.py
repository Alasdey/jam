from random import randint
import random

from config import ExperimentConfig
from creation.base import Creator, Program
from creation.genetics import TREE_MUTATION_OPS, TREE_CROSSOVER_OPS
from creation.homoiconic import HOMOICONIC_OPS


def gen_tree(n: int = 3) -> list[int]:
    """Uniform random plane tree with n nodes."""
    if n == 1:
        return Node()

    # Step 1: Generate a random sequence of (n-1) U's and (n-1) D's
    steps = [1] * (n - 1) + [-1] * (n - 1)
    random.shuffle(steps)

    # Step 2: Find the cyclic rotation that gives a valid Dyck path.
    # By the cycle lemma, the unique valid rotation starts just after
    # the position of the overall minimum (last occurrence, to be safe).
    min_val, min_idx = 0, 0
    height = 0
    for i, s in enumerate(steps):
        height += s
        if height <= min_val:          # use <= to get the last minimum
            min_val = height
            min_idx = i

    # Dyck word by cycling to the found unique correct position
    dyck = steps[min_idx + 1:] + steps[:min_idx + 1]

    # Converting to the 0 1 format
    res = [1 if i==1 else 0 for i in dyck]

    return res

"""
def derelict_gen_tree(n: int = 3) -> list[int]:
    res = [1]
    space = 0
    dept = space
    for _ in range(n - 1):
        space = randint(1, space + 1)
        res += [0] * (dept + 1 - space)
        res += [1]
        dept = space
    res += [0] * (dept + 1)
    return res
"""

class TreemoCreator(Creator):

    def __init__(self, cfg: ExperimentConfig):
        self._tree_size = cfg.treemo.tree_size
        self._rate = cfg.genetics.mutation_rate
        gc = cfg.genetics
        self._mutate_fn = TREE_MUTATION_OPS[gc.tree_mutation_op]
        self._crossover_fn = TREE_CROSSOVER_OPS[gc.tree_crossover_op]
        self._homoiconic_fn = HOMOICONIC_OPS["output"]

    def random(self) -> Program:
        return gen_tree(self._tree_size)

    def mutate(self, prog: Program) -> Program:
        return self._mutate_fn(prog, self._rate)

    def crossover(self, prog_a: Program, prog_b: Program) -> Program:
        return self._crossover_fn(prog_a, prog_b)

    def homoiconic(self, interp, prog_a: Program, prog_b: Program) -> Program:
        return self._homoiconic_fn(interp, prog_a, prog_b)

import random

from config import ExperimentConfig
from creation.base import Creator, Program
from creation.genetics import CODE_MUTATION_OPS, CODE_CROSSOVER_OPS
from creation.homoiconic import HOMOICONIC_OPS


class SubleqCreator(Creator):

    def __init__(self, cfg: ExperimentConfig):
        self._cfg = cfg
        self._length = cfg.code.code_length
        self._min = cfg.code.min_val
        self._max = cfg.code.max_val
        self._rate = cfg.genetics.mutation_rate
        gc = cfg.genetics
        self._mutate_fn = CODE_MUTATION_OPS[gc.code_mutation_op]
        self._crossover_fn = CODE_CROSSOVER_OPS[gc.code_crossover_op]
        self._homoiconic_fn = HOMOICONIC_OPS["output"]

    def random(self) -> Program:
        return [random.randint(self._min, self._max) for _ in range(self._length)]

    def mutate(self, prog: Program) -> Program:
        return self._mutate_fn(prog, self._cfg, self._rate)

    def crossover(self, prog_a: Program, prog_b: Program) -> Program:
        return self._crossover_fn(prog_a, prog_b)

    def homoiconic(self, interp, prog_a: Program, prog_b: Program) -> Program:
        return self._homoiconic_fn(interp, prog_a, prog_b)

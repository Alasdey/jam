from config import ExperimentConfig
from creation.base import Creator, Program
from interpreters.treemo_python.treemo_python import gen_tree
from creation.genetics import TREE_MUTATION_OPS, TREE_CROSSOVER_OPS
from creation.homoiconic import HOMOICONIC_OPS


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

import random

from config import GeneticsConfig
from creation.base import Creator


def make_offspring(
    creator: Creator,
    survivors: list,
    n_offspring: int,
    gc: GeneticsConfig,
    interp=None,
) -> tuple[list, list[str]]:
    if not survivors or n_offspring == 0:
        return [], []
    offspring = []
    methods = []
    for _ in range(n_offspring):
        if random.random() < gc.crossover_prob and len(survivors) >= 2:
            pa, pb = random.sample(survivors, 2)
            child = None
            if interp is not None and random.random() < gc.homoiconic_prob:
                child = creator.homoiconic(interp, pa, pb)
            if child is None:
                child = creator.crossover(pa, pb)
                methods.append("crossover")
            else:
                methods.append("homoiconic")
        else:
            parent = random.choice(survivors)
            child = creator.mutate(parent)
            methods.append("mutate")
        offspring.append(child)
    return offspring, methods

import random

from config import GeneticsConfig
from creation.base import Creator


def make_offspring(
    creator: Creator,
    survivors: list,
    n_offspring: int,
    gc: GeneticsConfig,
    interp=None,
) -> tuple[list, list[str], list[list[int]]]:
    """
    Returns (offspring, methods, parent_indices) where parent_indices[i] is
    the list of indices into survivors used to create offspring[i].
    """
    if not survivors or n_offspring == 0:
        return [], [], []
    offspring = []
    methods = []
    parent_indices = []
    for _ in range(n_offspring):
        if random.random() < gc.crossover_prob and len(survivors) >= 2:
            idxs = random.sample(range(len(survivors)), 2)
            pa, pb = survivors[idxs[0]], survivors[idxs[1]]
            child = None
            if interp is not None and random.random() < gc.homoiconic_prob:
                child = creator.homoiconic(interp, pa, pb)
            if child is None:
                child = creator.crossover(pa, pb)
                methods.append("crossover")
            else:
                methods.append("homoiconic")
            parent_indices.append(idxs)
        else:
            idx = random.randrange(len(survivors))
            child = creator.mutate(survivors[idx])
            methods.append("mutate")
            parent_indices.append([idx])
        offspring.append(child)
    return offspring, methods, parent_indices

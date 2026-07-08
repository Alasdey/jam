from typing import Callable, Union

import numpy as np

from config import CapStepConfig, NashStepConfig, SkimStepConfig
from selection.skim import iterated_elimination_strictly_dominated_rows_fast

# A selection step maps a square self-play payoff matrix to the sorted index
# array of surviving individuals; the loop applies it to pop and payoff.
SelectFn = Callable[[np.ndarray], np.ndarray]

StepConfig = Union[SkimStepConfig, CapStepConfig, NashStepConfig]


def build_skim(cfg: SkimStepConfig) -> SelectFn:
    def select(payoff: np.ndarray) -> np.ndarray:
        active = np.arange(payoff.shape[0])
        for _ in range(cfg.n_rounds):
            surviving = iterated_elimination_strictly_dominated_rows_fast(
                payoff[np.ix_(active, active)]
            )
            if cfg.fraction < 1.0:
                dominated = np.setdiff1d(np.arange(len(active)), surviving)
                n_keep = int(len(dominated) * (1 - cfg.fraction))
                kept = np.random.choice(dominated, size=n_keep, replace=False)
                surviving = np.sort(np.concatenate([surviving, kept]))
            active = active[surviving]
            if cfg.n_accepted and len(active) < cfg.n_accepted:
                break
        return active

    return select


def build_cap_top(cfg: CapStepConfig) -> SelectFn:
    def select(payoff: np.ndarray) -> np.ndarray:
        n = payoff.shape[0]
        if n <= cfg.max_pop:
            return np.arange(n)
        scores = payoff.sum(axis=1)
        return np.sort(np.argsort(scores)[-cfg.max_pop:])

    return select


def build_cap_random(cfg: CapStepConfig) -> SelectFn:
    def select(payoff: np.ndarray) -> np.ndarray:
        n = payoff.shape[0]
        if n <= cfg.max_pop:
            return np.arange(n)
        return np.sort(np.random.choice(n, size=cfg.max_pop, replace=False))

    return select


def build_nash(cfg: NashStepConfig) -> SelectFn:
    # support_enumeration is exponential in population size: small pops only.
    from selection.nash_set import compute_nash_subset

    def select(payoff: np.ndarray) -> np.ndarray:
        return np.array(compute_nash_subset(payoff), dtype=int)

    return select


SELECTION_STEPS: dict[str, Callable[..., SelectFn]] = {
    "skim": build_skim,
    "cap_top": build_cap_top,
    "cap_random": build_cap_random,
    "nash": build_nash,
}


def build_selection(steps: list[StepConfig]) -> list[SelectFn]:
    return [SELECTION_STEPS[s.kind](s) for s in steps]

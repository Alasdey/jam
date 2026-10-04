from typing import Callable, Optional, Sequence, Union

import numpy as np

from config import CapStepConfig, DedupeStepConfig, MaxLengthStepConfig, NashStepConfig, SkimStepConfig, LexicaseStepConfig
from core.types import Program
from selection.lexicase import lexicase_selection
from config import KNNNoveltyStepConfig
from selection.knn_novelty import knn_novelty_selection
from selection.skim import iterated_elimination_strictly_dominated_rows_fast

# A selection step maps the square self-play payoff matrix, plus the genomes of
# the individuals it is indexed by, to the sorted index array of survivors; the
# loop applies the result to pop and payoff. Genomes are passed because some
# criteria are not functions of the payoff matrix at all: `dedupe` compares
# programs, and payoff rows cannot stand in for them (a reward that draws every
# matchup makes every row identical). Steps that only need the payoff ignore it.
#
# The same steps drive two pipelines. RunConfig.selection runs after evaluation
# and gets the real matrix; RunConfig.pre_selection runs before it and gets
# payoff=None, so the individuals it drops never cost a matchup. Only the steps
# named in PAYOFF_FREE_STEPS may be used pre-evaluation.
SelectFn = Callable[[Optional[np.ndarray], Sequence[Program]], np.ndarray]

StepConfig = Union[DedupeStepConfig, MaxLengthStepConfig, SkimStepConfig, CapStepConfig, NashStepConfig, LexicaseStepConfig, KNNNoveltyStepConfig]

# Steps that never read the payoff matrix, and so can run pre-evaluation.
PAYOFF_FREE_STEPS = frozenset({"dedupe", "max_length"})


def build_dedupe(cfg: DedupeStepConfig) -> SelectFn:
    def select(payoff: Optional[np.ndarray], genomes: Sequence[Program]) -> np.ndarray:
        seen: set[tuple[int, ...]] = set()
        keep: list[int] = []
        for i, genome in enumerate(genomes):
            key = tuple(genome)
            if key not in seen:
                seen.add(key)
                keep.append(i)
        return np.array(keep, dtype=int)

    return select


def build_max_length(cfg: MaxLengthStepConfig) -> SelectFn:
    if cfg.max_length < 0:
        raise ValueError("max_length must be non-negative")

    def select(payoff: Optional[np.ndarray], genomes: Sequence[Program]) -> np.ndarray:
        return np.array(
            [i for i, genome in enumerate(genomes) if len(genome) <= cfg.max_length],
            dtype=int,
        )

    return select


def build_skim(cfg: SkimStepConfig) -> SelectFn:
    def select(payoff: np.ndarray, genomes: Sequence[Program]) -> np.ndarray:
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


def build_lexicase(cfg: LexicaseStepConfig) -> SelectFn:
    def select(payoff: np.ndarray, genomes: Sequence[Program]) -> np.ndarray:
        return lexicase_selection(payoff, cfg.n_accepted)

    return select


def build_knn_novelty(cfg: KNNNoveltyStepConfig) -> SelectFn:
    def select(payoff: np.ndarray, genomes: Sequence[Program]) -> np.ndarray:
        return knn_novelty_selection(payoff, cfg.n_accepted, cfg.k)

    return select


def build_cap_top(cfg: CapStepConfig) -> SelectFn:
    def select(payoff: np.ndarray, genomes: Sequence[Program]) -> np.ndarray:
        n = payoff.shape[0]
        if n <= cfg.max_pop:
            return np.arange(n)
        scores = payoff.sum(axis=1)
        return np.sort(np.argsort(scores)[-cfg.max_pop:])

    return select


def build_cap_random(cfg: CapStepConfig) -> SelectFn:
    def select(payoff: np.ndarray, genomes: Sequence[Program]) -> np.ndarray:
        n = payoff.shape[0]
        if n <= cfg.max_pop:
            return np.arange(n)
        return np.sort(np.random.choice(n, size=cfg.max_pop, replace=False))

    return select


def build_nash(cfg: NashStepConfig) -> SelectFn:
    # support_enumeration is exponential in population size: small pops only.
    from selection.nash_set import compute_nash_subset

    def select(payoff: np.ndarray, genomes: Sequence[Program]) -> np.ndarray:
        return np.array(compute_nash_subset(payoff), dtype=int)

    return select


SELECTION_STEPS: dict[str, Callable[..., SelectFn]] = {
    "dedupe": build_dedupe,
    "max_length": build_max_length,
    "skim": build_skim,
    "lexicase": build_lexicase,
    "knn_novelty": build_knn_novelty,
    "cap_top": build_cap_top,
    "cap_random": build_cap_random,
    "nash": build_nash,
}


def build_selection(steps: list[StepConfig], pre: bool = False) -> list[SelectFn]:
    """Compose steps into a pipeline. `pre=True` for the pre-evaluation one.

    Pre-evaluation steps are called with payoff=None, so a step that reads the
    matrix is rejected here rather than failing mid-run.
    """
    if pre:
        offenders = [s.kind for s in steps if s.kind not in PAYOFF_FREE_STEPS]
        if offenders:
            raise ValueError(
                f"pre_selection steps run before the payoff matrix exists, so only "
                f"{sorted(PAYOFF_FREE_STEPS)} may be used there; got {offenders}"
            )
    return [SELECTION_STEPS[s.kind](s) for s in steps]

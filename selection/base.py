from typing import Callable, Optional, Union

import numpy as np

from config import (
    BestWorstRandomStepConfig, CapStepConfig, DedupeStepConfig, KNNNoveltyStepConfig, KNNScoreStepConfig,
    LexicaseStepConfig, MaxLengthStepConfig, MixtureStepConfig, NashStepConfig, SkimStepConfig,
)
from core.device import cupy, loud_warning
from core.population import Population
from core.rng import DeviceRng
from selection.best_worst_random import best_worst_random_selection
from selection.knn_novelty import knn_novelty_selection, rank_descending
from selection.knn_score import knn_score_selection
from selection.lexicase import lexicase_selection
from selection.mixture import mixture_selection
from selection.skim import iterated_elimination

# A selection step maps the square self-play payoff matrix (device int32),
# the population it is indexed by, and the generation's random numbers to
# the sorted indices of the survivors (a device or host int array); the loop
# applies them to the population and the matrix. The population is passed
# because some criteria are not functions of the payoff at all: `dedupe`
# compares genomes, which payoff rows cannot stand in for (a reward that
# draws every matchup makes every row identical). Steps that only need the
# payoff ignore it.
#
# The same steps drive two pipelines. RunConfig.selection runs after
# evaluation and gets the real matrix; RunConfig.pre_selection runs before it
# and gets payoff=None, so the individuals it drops never cost a matchup.
# Only the steps named in PAYOFF_FREE_STEPS may be used pre-evaluation.
#
# To add a step: a config dataclass in config.py with a `kind`, a builder
# returning a SelectFn below, and an entry in SELECTION_STEPS. Write it with
# CuPy (the NumPy API on the device) and draw randomness from `rng`
# (core/rng.py), so it runs on the GPU and replays exactly on resume.
SelectFn = Callable[[Optional[object], Population, DeviceRng], object]

StepConfig = Union[DedupeStepConfig, MaxLengthStepConfig, SkimStepConfig, CapStepConfig, NashStepConfig,
                   LexicaseStepConfig, KNNNoveltyStepConfig, KNNScoreStepConfig, BestWorstRandomStepConfig,
                   MixtureStepConfig]

# Steps that never read the payoff matrix, and so can run pre-evaluation.
PAYOFF_FREE_STEPS = frozenset({"dedupe", "max_length"})


def build_dedupe(cfg: DedupeStepConfig) -> SelectFn:
    def select(payoff, pop: Population, rng: DeviceRng):
        cp = cupy()
        n = len(pop)
        if n == 0:
            return cp.zeros(0, dtype=cp.int64)
        index = cp.arange(n, dtype=cp.int64)
        hashes = pop.hashes()
        order = cp.lexsort(cp.stack([index, hashes.astype(cp.int64)]))
        h = hashes[order]
        start = cp.ones(n, dtype=bool)
        start[1:] = h[1:] != h[:-1]
        leader = order[cp.flatnonzero(start)][cp.cumsum(start) - 1]  # first index of each equal-hash run
        # Duplicates of the run's first genome go; a hash collision (equal
        # hash, different genome) is kept.
        duplicate = pop.equal(leader, order) & (leader != order)
        keep = cp.ones(n, dtype=bool)
        keep[order[duplicate]] = False
        return cp.flatnonzero(keep).astype(cp.int64)

    return select


def build_max_length(cfg: MaxLengthStepConfig) -> SelectFn:
    if cfg.max_length < 0:
        raise ValueError("max_length must be non-negative")

    def select(payoff, pop: Population, rng: DeviceRng):
        cp = cupy()
        return cp.flatnonzero(pop.length <= cfg.max_length).astype(cp.int64)

    return select


def build_skim(cfg: SkimStepConfig) -> SelectFn:
    def select(payoff, pop: Population, rng: DeviceRng):
        cp = cupy()
        active = cp.arange(payoff.shape[0], dtype=cp.int64)
        for _ in range(cfg.n_rounds):
            surviving = iterated_elimination(payoff[cp.ix_(active, active)])
            if cfg.fraction < 1.0:
                dominated = cp.flatnonzero(~cp.isin(cp.arange(len(active)), surviving))
                n_keep = int(len(dominated) * (1 - cfg.fraction))
                kept = dominated[rng.permutation(len(dominated))[:n_keep]]
                surviving = cp.sort(cp.concatenate([surviving, kept]))
            active = active[surviving]
            if cfg.n_accepted and len(active) < cfg.n_accepted:
                break
        return active

    return select


def build_lexicase(cfg: LexicaseStepConfig) -> SelectFn:
    def select(payoff, pop: Population, rng: DeviceRng):
        return lexicase_selection(payoff, cfg.n_accepted, rng)

    return select


def build_knn_novelty(cfg: KNNNoveltyStepConfig) -> SelectFn:
    def select(payoff, pop: Population, rng: DeviceRng):
        return knn_novelty_selection(payoff, cfg.n_accepted, cfg.k, rng)

    return select


def build_knn_score(cfg: KNNScoreStepConfig) -> SelectFn:
    def select(payoff, pop: Population, rng: DeviceRng):
        return knn_score_selection(payoff, cfg.n_accepted, cfg.k, rng)

    return select


def build_best_worst_random(cfg: BestWorstRandomStepConfig) -> SelectFn:
    def select(payoff, pop: Population, rng: DeviceRng):
        return best_worst_random_selection(payoff, cfg.n_best, cfg.n_worst, cfg.n_rand, rng)

    return select


def build_mixture(cfg: MixtureStepConfig) -> SelectFn:
    def select(payoff, pop: Population, rng: DeviceRng):
        return mixture_selection(payoff, cfg.n_best, cfg.n_good, cfg.n_good_novel, cfg.n_novelty, cfg.k, rng)

    return select


def build_cap_top(cfg: CapStepConfig) -> SelectFn:
    """Keep the max_pop best payoff row-sums; ties broken randomly."""
    def select(payoff, pop: Population, rng: DeviceRng):
        cp = cupy()
        n = payoff.shape[0]
        if n <= cfg.max_pop:
            return cp.arange(n, dtype=cp.int64)
        return cp.sort(rank_descending(payoff.sum(axis=1, dtype=cp.int64), rng)[:cfg.max_pop])

    return select


def build_cap_random(cfg: CapStepConfig) -> SelectFn:
    def select(payoff, pop: Population, rng: DeviceRng):
        cp = cupy()
        n = payoff.shape[0]
        if n <= cfg.max_pop:
            return cp.arange(n, dtype=cp.int64)
        return cp.sort(rng.permutation(n)[:cfg.max_pop])

    return select


def build_nash(cfg: NashStepConfig) -> SelectFn:
    # Support enumeration (nashpy) is exponential in population size and runs
    # on the CPU: small populations only.
    from selection.nash_set import compute_nash_subset

    def select(payoff, pop: Population, rng: DeviceRng):
        cp = cupy()
        loud_warning(f"nash selection copies the {payoff.shape[0]} x {payoff.shape[1]} payoff matrix to the "
                     "host and enumerates supports with nashpy (exponential in population size)")
        return cp.asarray(np.array(compute_nash_subset(cp.asnumpy(payoff)), dtype=np.int64))

    return select


SELECTION_STEPS: dict[str, Callable[..., SelectFn]] = {
    "dedupe": build_dedupe,
    "max_length": build_max_length,
    "skim": build_skim,
    "lexicase": build_lexicase,
    "knn_novelty": build_knn_novelty,
    "knn_score": build_knn_score,
    "best_worst_random": build_best_worst_random,
    "mixture": build_mixture,
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

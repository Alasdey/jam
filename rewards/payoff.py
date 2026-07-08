from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List

import numpy as np

from config import ExperimentConfig
from core.types import Program
from interpreters import build_interpreter
from rewards.base import RewardSpec

# These will be "per-process" globals in worker processes
_INTERP = None
_REWARD_FN: Callable | None = None


def _init_worker(cfg: ExperimentConfig, reward_fn: Callable):
    """
    Called once in each worker process.
    Builds the interpreter and stores the reward function as per-process globals.
    """
    global _INTERP, _REWARD_FN
    _INTERP = build_interpreter(cfg)  # interpreter built ONCE per worker
    _REWARD_FN = reward_fn            # top-level function, picklable


def _compute_single_matchup(args):
    """
    Worker function that computes reward for a single (i, j) matchup.
    Uses per-process globals _INTERP and _REWARD_FN.
    """
    i, j, code_a, code_b = args
    reward = _REWARD_FN(_INTERP, code_a, code_b)  # type: ignore[misc]
    return i, j, reward


class PayoffEngine:
    """
    Single choke point for matchup evaluation, shared by the evolution loop
    and the offline tournament tool.

    With n_workers > 1 the ProcessPoolExecutor is created once and reused
    across calls; future matchup caching or alternative backends slot into
    matrix() without touching callers.
    """

    def __init__(self, exp_cfg: ExperimentConfig, reward: RewardSpec):
        self.exp_cfg = exp_cfg
        self.reward = reward
        self._n_workers = exp_cfg.payoff.n_workers
        self._interp = None
        self._executor = None
        if self._n_workers > 1:
            self._executor = ProcessPoolExecutor(
                max_workers=self._n_workers,
                initializer=_init_worker,
                initargs=(exp_cfg, reward.fn),
            )
        else:
            self._interp = build_interpreter(exp_cfg)

    def matrix(self, ref: List[Program], pop: List[Program]) -> np.ndarray:
        """
        Payoff matrix of shape (len(ref), len(pop)); entry [i, j] is the reward
        for ref[i] playing against pop[j].
        """
        payoff = np.zeros((len(ref), len(pop)), dtype=int)

        if self._executor is None:
            for i, code_a in enumerate(ref):
                for j, code_b in enumerate(pop):
                    payoff[i, j] = self.reward.fn(self._interp, code_a, code_b)
            return payoff

        matchups = [
            (i, j, ref[i], pop[j])
            for i in range(len(ref))
            for j in range(len(pop))
        ]
        chunksize = max(1, len(matchups) // (self._n_workers * 8))
        for i, j, r in self._executor.map(_compute_single_matchup, matchups, chunksize=chunksize):
            payoff[i, j] = r
        return payoff

    def extend(self, payoff: np.ndarray, old: List[Program], new: List[Program]) -> np.ndarray:
        """
        Extend a square self-play payoff matrix over `old` to cover old + new.
        The reverse block is derived as -old_new.T when the reward is zero-sum,
        computed explicitly otherwise.
        """
        new_new = self.matrix(new, new)
        if not old:
            return new_new
        old_new = self.matrix(old, new)
        new_old = -old_new.T if self.reward.zero_sum else self.matrix(new, old)
        return np.block([
            [payoff, old_new],
            [new_old, new_new],
        ])

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __enter__(self) -> "PayoffEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


import numpy as np
from typing import List, Callable
from concurrent.futures import ProcessPoolExecutor

from config import ExperimentConfig
from interpreters.wrapper import make_interpreter
from rewards.wrapper import make_reward


# These will be "per-process" globals in worker processes
_INTERP = None
_REWARD_FN: Callable | None = None


def _init_worker(cfg: ExperimentConfig, reward_fn: Callable):
    """
    Called once in each worker process.
    Builds the interpreter and stores the reward function as per-process globals.
    """
    global _INTERP, _REWARD_FN
    _INTERP = make_interpreter(cfg)     # intesrpreter built ONCE per worker
    _REWARD_FN = make_reward(cfg)       # top-level function, picklable


def _compute_single_matchup(args):
    """
    Worker function that computes reward for a single (i, j) matchup.
    Uses per-process globals _INTERP and _REWARD_FN.
    """
    global _INTERP, _REWARD_FN
    i, j, code_a, code_b = args
    reward = _REWARD_FN(_INTERP, code_a, code_b)  # type: ignore[arg-type]
    return i, j, reward


def compute_payoff_matrix(
    cfg: ExperimentConfig,
    ref: List[List[int]],
    pop: List[List[int]],
    reward_fn: Callable,
) -> np.ndarray:
    """
    Compute payoff matrix for ref x pop using given reward_fn and interpreter from cfg.

    Args:
        cfg: ExperimentConfig with interpreter + payoff settings.
        ref: Reference population (list of programs).
        pop: Population to evaluate (list of programs).
        reward_fn: reward(interp, code_a, code_b) -> int

    Returns:
        Payoff matrix of shape (len(ref), len(pop)) where entry [i, j] is the reward
        for ref[i] playing against pop[j]
    """
    n_ref = len(ref)
    n_pop = len(pop)
    payoff_matrix = np.zeros((n_ref, n_pop), dtype=int)

    n_workers = cfg.payoff.n_workers

    # --- Sequential path (simpler, good for debugging) ---
    if n_workers == 1:
        interp = make_interpreter(cfg)
        for i in range(n_ref):
            for j in range(n_pop):
                payoff_matrix[i, j] = reward_fn(interp, ref[i], pop[j])
        return payoff_matrix
    
    # --- Parallel path ---
    matchups = [
        (i, j, ref[i], pop[j])
        for i in range(n_ref)
        for j in range(n_pop)
    ]

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(cfg, reward_fn),
    ) as executor:
        for i, j, r in executor.map(_compute_single_matchup, matchups):
            payoff_matrix[i, j] = r

    return payoff_matrix

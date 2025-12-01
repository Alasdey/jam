import numpy as np
from typing import List, Callable
from interpreters.subleq.subleq import subleq
from concurrent.futures import ProcessPoolExecutor


def _compute_single_matchup(args):
    """Helper function for parallel computation of a single matchup."""
    i, j, code_a, code_b, reward_fn = args
    interp = subleq()
    reward = reward_fn(interp, code_a, code_b)
    return (i, j, reward)


def compute_payoff_matrix(
    ref: List[List[int]], 
    pop: List[List[int]],
    reward_fn: Callable[[subleq, List[int], List[int]], int],
    n_workers: int = 1
) -> np.ndarray:
    """
    Compute the payoff matrix for two populations in a symmetric zero-sum game.
    
    Args:
        ref: Reference population, list of SUBLEQ programs (each program is a List[int])
        pop: Current population, list of SUBLEQ programs
        reward_fn: Reward function that takes (interpreter, code_a, code_b) and returns reward for a
        n_workers: Number of worker processes (1 = sequential, >1 = parallel)
        
    Returns:
        Payoff matrix of shape (len(ref), len(pop)) where entry [i, j] is the reward
        for ref[i] playing against pop[j]
    """
    n_ref = len(ref)
    n_pop = len(pop)
    payoff_matrix = np.zeros((n_ref, n_pop), dtype=int)
    
    if n_workers == 1:
        # Sequential computation with single interpreter instance
        interp = subleq()
        for i in range(n_ref):
            for j in range(n_pop):
                payoff_matrix[i, j] = reward_fn(interp, ref[i], pop[j])
    else:
        # Parallel computation
        matchups = [
            (i, j, ref[i], pop[j], reward_fn)
            for i in range(n_ref)
            for j in range(n_pop)
        ]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = executor.map(_compute_single_matchup, matchups)
            
            for i, j, reward in results:
                payoff_matrix[i, j] = reward
    
    return payoff_matrix
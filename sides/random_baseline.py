
import json
import time

from utils.pop_init import random_code, random_population
from rewards.payoff import compute_payoff_matrix
from rewards.reward import reward


def main(
    n_ref: int, 
    n_tested: int, 
    code_length: int, 
    min_val: int, 
    max_val: int, 
    max_iter: int,
    max_out: int, 
    out_path: str,
    n_worker: int,
    ):
    """
    """
    # reward_fn = lambda interp, code_a, code_b: reward(interp, code_a, code_b, max_iter, max_out)
    ref_pop = random_population(
        n_ref, 
        code_length, 
        min_val, 
        max_val
    )
    file = open(out_path, "a")
    for _ in range(n_tested):
        indiv = random_code(
            code_length,
            min_val,
            max_val,
        )
        payoff = compute_payoff_matrix(
            ref_pop,
            [indiv],
            reward,
        )
        score = sum(payoff[0])
        json.dump(score)
    return


if __name__ == "__main__":
    n_tested = 10**7
    n_ref = 10**3
    code_length = 500
    min_val = -500
    max_val = 1000
    max_iter = 20000
    max_out = 2000
    out_path = "outputs/random_baseline/"
    n_worker = 7

    out_path += time.strftime("%Y%m%d_%H%M%S")
    out_path += ".json"

    main(
        n_ref, 
        n_tested, 
        code_length, 
        min_val, 
        max_val, 
        max_iter, 
        max_out,
        out_path,
        n_worker,
    )
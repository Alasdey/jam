

from pathlib import Path
import numpy as np
import time
import json
from typing import Optional

from selection.nash_set import compute_nash_equilibrium, compute_nash_subset
from selection.skim import iterated_elimination_strictly_dominated_rows, iterated_elimination_strictly_dominated_rows_fast
from rewards.payoff import compute_payoff_matrix
from utils.pop_init import random_population
from config import ExperimentConfig
from rewards.wrapper import make_reward


def main(
        n_pop: int=100, 
        n_iter: int=1000, 
        n_skim: int=1, 
        n_accepted: Optional[int]=None,
        out_dir: str="outputs/test"
    ):
    cfg = ExperimentConfig()
    reward_fn = make_reward(cfg)
    pop = []
    payoff = compute_payoff_matrix(cfg, pop, pop, reward_fn)
    for i in range(n_iter):
        n_old = len(pop)
        pop += random_population(
            n_pop, 
            cfg.code.code_length, 
            cfg.code.min_val, 
            cfg.code.max_val,
        )
        print(f"Starting gen {i}, with population {len(pop)}")
        t0 = time.time()
        payoff_new_new = compute_payoff_matrix(cfg, pop[n_old:], pop[n_old:], reward_fn)
        if n_old == 0:
            payoff = payoff_new_new
        else:
            payoff_old_new = compute_payoff_matrix(cfg, pop[:n_old], pop[n_old:], reward_fn)
            payoff = np.append(payoff, payoff_old_new, axis=1)
            temp = np.append(-payoff_old_new.T, payoff_new_new, axis=1)
            payoff = np.append(payoff, temp, axis=0)
        t1 = time.time()
        n_prev = n_old
        for skim in range(n_skim):
            skimmed = iterated_elimination_strictly_dominated_rows_fast(payoff)
            n_removed = n_old - int((skimmed < n_old).sum())
            n_new = int((skimmed >= n_old).sum())
            pop = [pop[i] for i in skimmed.tolist()]
            payoff = payoff[skimmed, :][:, skimmed]
            n_old = n_old - n_removed
            if n_accepted and len(pop) < n_accepted:
                break
        t2 = time.time()
        print(f"Removed {n_prev-n_old}, New {len(pop)-n_old}, took {t1 - t0} and {t2 - t1}")
        with open(out_dir + "/pop.json", "w") as f:
            json.dump(pop, f)
            f.write("\n")
        with open(out_dir + "/logs.json", "a") as f:
            json.dump(len(pop), f)
            f.write("\n")

if __name__ == "__main__":
    n_skim = 2
    n_pop = 100
    n_iter = 10**7
    n_accepted = 2000
    out_dir_path = "outputs/random_skimmed/" + time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    main(n_pop=n_pop, n_iter=n_iter, n_skim=n_skim, n_accepted=n_accepted, out_dir=out_dir_path)

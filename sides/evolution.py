import time
from dataclasses import asdict

import numpy as np

from config import EvolutionConfig
from creation.factory import make_creator
from creation.offspring import make_offspring
from interpreters.wrapper import make_interpreter
from loggers import ExperimentLogger
from rewards.payoff import compute_payoff_matrix
from rewards.wrapper import make_reward
from selection.nash_set import compute_nash_subset
from selection.skim import (
    iterated_elimination_strictly_dominated_rows,
    iterated_elimination_strictly_dominated_rows_fast,
)


def _select(payoff: np.ndarray, method: str, n_skim: int) -> np.ndarray:
    """Return a sorted index array of survivors from the current payoff matrix."""
    if method == "none":
        return np.arange(payoff.shape[0])

    if method in ("skim_fast", "skim_slow"):
        skim_fn = (
            iterated_elimination_strictly_dominated_rows_fast
            if method == "skim_fast"
            else iterated_elimination_strictly_dominated_rows
        )
        active = np.arange(payoff.shape[0])
        for _ in range(n_skim):
            local = skim_fn(payoff[np.ix_(active, active)])
            active = active[local]
        return active

    if method == "nash_subset":
        indices = compute_nash_subset(payoff)
        return np.array(indices, dtype=int)

    raise ValueError(f"Unknown selection method: {method!r}")


def _expand_payoff(
    payoff: np.ndarray,
    pop: list,
    offspring: list,
    cfg,
    reward_fn,
) -> np.ndarray:
    """
    Expand a square self-play payoff matrix to include new offspring.

    Assumes zero-sum: payoff(new, old) = -payoff(old, new)^T.
    """
    pay_old_new = compute_payoff_matrix(cfg, pop, offspring, reward_fn)
    pay_new_new = compute_payoff_matrix(cfg, offspring, offspring, reward_fn)
    return np.block([
        [payoff,          pay_old_new],
        [-pay_old_new.T,  pay_new_new],
    ])


def main(cfg: EvolutionConfig) -> None:
    exp = cfg.experiment
    creator = make_creator(exp)
    reward_fn = make_reward(exp)
    interp = make_interpreter(exp)

    logger = ExperimentLogger(cfg.out_dir, cfg)

    pop = [creator.random() for _ in range(cfg.n_init)]
    payoff = compute_payoff_matrix(exp, pop, pop, reward_fn)

    for gen in range(cfg.n_iter):
        t0 = time.time()

        offspring = make_offspring(
            creator=creator,
            survivors=pop,
            n_offspring=cfg.n_offspring,
            gc=exp.genetics,
            interp=interp,
        )

        if offspring:
            payoff = _expand_payoff(payoff, pop, offspring, exp, reward_fn)
            pop = pop + offspring

        t1 = time.time()

        n_before = len(pop)
        survivors = _select(payoff, cfg.selection, cfg.n_skim)

        if cfg.pop_cap is not None and len(survivors) > cfg.pop_cap:
            survivors = survivors[: cfg.pop_cap]

        pop = [pop[i] for i in survivors.tolist()]
        payoff = payoff[np.ix_(survivors, survivors)]

        t2 = time.time()

        logger.log(
            {
                "gen": gen,
                "t": t2,
                "payoff_s": round(t1 - t0, 4),
                "selection_s": round(t2 - t1, 4),
                "pop_size": len(pop),
                "n_added": len(offspring),
                "n_removed": n_before - len(pop),
                "payoff_mean": round(float(payoff.mean()), 4) if payoff.size else 0,
                "payoff_std": round(float(payoff.std()), 4) if payoff.size else 0,
                "payoff_min": int(payoff.min()) if payoff.size else 0,
                "payoff_max": int(payoff.max()) if payoff.size else 0,
            },
            work_pop=pop,
        )

        print(
            f"gen {gen:>6} | pop {len(pop):>5} | "
            f"+{len(offspring)} -{n_before - len(pop)} | "
            f"payoff {t1 - t0:.2f}s  select {t2 - t1:.2f}s"
        )


if __name__ == "__main__":
    main(EvolutionConfig())

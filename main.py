from pathlib import Path
import numpy as np
import time

from selection.skim import iterated_elimination_strictly_dominated_rows_fast
from rewards.payoff import compute_payoff_matrix
from loggers import ExperimentLogger
from creation.factory import make_creator
from creation.offspring import make_offspring
from interpreters.wrapper import make_interpreter
from config import MainConfig
from rewards.wrapper import make_reward


def main(cfg: MainConfig):
    creator = make_creator(cfg.experiment)
    interp = make_interpreter(cfg.experiment)
    logger = ExperimentLogger(cfg.out_dir, cfg)
    reward_fn = make_reward(cfg.experiment)
    pop = []
    payoff = compute_payoff_matrix(cfg.experiment, pop, pop, reward_fn)
    for i in range(cfg.n_iter):
        n_old = len(pop)
        new_individuals = [creator.random() for _ in range(cfg.n_random)]
        if n_old > 0:
            new_individuals += make_offspring(
                creator=creator,
                survivors=pop,
                n_offspring=cfg.n_offspring,
                gc=cfg.experiment.genetics,
                interp=interp,
            )
        else:
            # Bootstrap: nothing to mutate/crossover from yet
            new_individuals += [creator.random() for _ in range(cfg.n_offspring)]
        pop += new_individuals
        print(f"Starting gen {i}, with population {len(pop)}")
        t0 = time.time()
        payoff_new_new = compute_payoff_matrix(cfg.experiment, pop[n_old:], pop[n_old:], reward_fn)
        if n_old == 0:
            payoff = payoff_new_new
        else:
            payoff_old_new = compute_payoff_matrix(cfg.experiment, pop[:n_old], pop[n_old:], reward_fn)
            payoff = np.append(payoff, payoff_old_new, axis=1)
            temp = np.append(-payoff_old_new.T, payoff_new_new, axis=1)
            payoff = np.append(payoff, temp, axis=0)

        payout = payoff.copy()
        t1 = time.time()
        n_prev = n_old
        for _ in range(cfg.n_skim):
            skimmed = iterated_elimination_strictly_dominated_rows_fast(payoff)
            if cfg.skim_fraction < 1.0:
                dominated = np.setdiff1d(np.arange(len(pop)), skimmed)
                n_keep = int(len(dominated) * (1 - cfg.skim_fraction))
                kept = np.random.choice(dominated, size=n_keep, replace=False)
                skimmed = np.sort(np.concatenate([skimmed, kept]))
            n_removed = n_old - int((skimmed < n_old).sum())
            n_new = int((skimmed >= n_old).sum())
            pop = [pop[i] for i in skimmed.tolist()]
            payoff = payoff[skimmed, :][:, skimmed]
            n_old = n_old - n_removed
            if cfg.n_accepted and len(pop) < cfg.n_accepted:
                break
        n_capped = 0
        if cfg.max_pop and len(pop) > cfg.max_pop:
            n_capped = len(pop) - cfg.max_pop
            scores = payoff.sum(1)
            keep = np.sort(np.argsort(scores)[-cfg.max_pop:])
            pop = [pop[i] for i in keep]
            payoff = payoff[keep, :][:, keep]
            n_old = int((keep < n_old).sum())
        t2 = time.time()
        n_removed_total = n_prev - n_old
        n_survived_new = len(pop) - n_old
        print(f"Removed {n_removed_total} ({n_capped} via max_pop cap), New {n_survived_new}, took {t1 - t0:.2f}s payoff, {t2 - t1:.2f}s skim")
        payoff_flat = payoff.sum(1)
        payout_flat = payout.sum(1)
        logger.log(
            {
                "gen": i,
                "t": t2,
                "payoff_s": round(t1 - t0, 4),
                "skim_s": round(t2 - t1, 4),
                "pop_size": len(pop),
                "n_added": len(new_individuals),
                "n_removed": n_removed_total,
                "n_capped": n_capped,
                "n_survived_new": n_survived_new,
                "payoff_mean": round(float(payoff_flat.mean()), 4),
                "payoff_std": round(float(payoff_flat.std()), 4),
                "payoff_min": int(payoff_flat.min()),
                "payoff_max": int(payoff_flat.max()),
                "payout_mean": round(float(payout_flat.mean()), 4),
                "payout_std": round(float(payout_flat.std()), 4),
                "payout_min": int(payout_flat.min()),
                "payout_max": int(payout_flat.max()),
            },
            work_pop=pop,
        )


if __name__ == "__main__":
    cfg = MainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    main(cfg)

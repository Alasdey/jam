import json
from collections import Counter
from pathlib import Path
import numpy as np
import time

from selection.skim import skim_population
from rewards.payoff import compute_payoff_matrix
from loggers import ExperimentLogger
from creation.factory import make_creator
from creation.offspring import make_offspring
from interpreters.wrapper import make_interpreter
from config import MainConfig
from rewards.wrapper import make_reward


def _last_gen(metrics_path: Path) -> int:
    last = -1
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)["gen"]
    return last


def load_population(expe_dir: str) -> tuple[list, list[dict]]:
    """
    Returns (genomes, meta) where meta[i] = {"id": int, "method": str|None, "parents": list[int]}.
    Handles both new dict format and old raw-list format (assigns sequential IDs).
    """
    pops_dir = Path(expe_dir) / "populations"
    work_pops = sorted(pops_dir.glob("work_pop_*.json"))
    if not work_pops:
        raise FileNotFoundError(f"No work_pop_*.json files found in {pops_dir}")
    latest = work_pops[-1]
    genomes = []
    meta = []
    with open(latest) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                # old raw-genome format
                genomes.append(obj)
                meta.append({"id": i, "method": None, "parents": []})
            else:
                genomes.append(obj["genome"])
                meta.append({"id": obj["id"], "method": obj.get("method"), "parents": obj.get("parents", [])})
    print(f"Resuming from {latest} with population {len(genomes)}")
    return genomes, meta


def main(cfg: MainConfig):
    creator = make_creator(cfg.experiment)
    interp = make_interpreter(cfg.experiment)
    logger = ExperimentLogger(cfg.out_dir, cfg)
    reward_fn = make_reward(cfg.experiment)

    pop = []
    pop_meta = []  # parallel list of {"id": int, "method": str|None, "parents": list[int]}
    next_id = 0
    gen_offset = 0

    if cfg.resume_from:
        pop, pop_meta = load_population(cfg.resume_from)
        if pop_meta:
            next_id = max(m["id"] for m in pop_meta) + 1
        metrics_path = Path(cfg.out_dir) / "metrics.jsonl"
        if metrics_path.exists():
            gen_offset = _last_gen(metrics_path) + 1

    payoff = compute_payoff_matrix(cfg.experiment, pop, pop, reward_fn)

    for i in range(cfg.n_iter):
        n_old = len(pop)

        new_individuals = [creator.random() for _ in range(cfg.n_random)]
        new_methods = ["random"] * cfg.n_random
        new_parent_ids = [[] for _ in range(cfg.n_random)]

        if n_old > 0:
            offspring, offspring_methods, offspring_parent_indices = make_offspring(
                creator=creator,
                survivors=pop,
                n_offspring=cfg.n_offspring,
                gc=cfg.experiment.genetics,
                interp=interp,
            )
            new_individuals += offspring
            new_methods += offspring_methods
            for pidxs in offspring_parent_indices:
                new_parent_ids.append([pop_meta[j]["id"] for j in pidxs])
        else:
            bootstrap = [creator.random() for _ in range(cfg.n_offspring)]
            new_individuals += bootstrap
            new_methods += ["random"] * cfg.n_offspring
            new_parent_ids += [[] for _ in range(cfg.n_offspring)]

        method_counts = Counter(new_methods)

        new_ids = list(range(next_id, next_id + len(new_individuals)))
        next_id += len(new_individuals)

        new_meta = [
            {"id": nid, "method": m, "parents": p}
            for nid, m, p in zip(new_ids, new_methods, new_parent_ids)
        ]

        pop_meta = [{"id": m["id"], "method": m["method"], "parents": m["parents"]} for m in pop_meta] + new_meta
        pop += new_individuals

        print(f"Starting gen {gen_offset + i}, with population {len(pop)}")
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

        if cfg.skim_when in ("before", "both"):
            pop, pop_meta, payoff, n_old = skim_population(
                pop, pop_meta, payoff, n_old, cfg.n_skim, cfg.skim_fraction, cfg.n_accepted,
            )

        n_capped = 0
        if cfg.max_pop and len(pop) > cfg.max_pop:
            n_capped = len(pop) - cfg.max_pop
            scores = payoff.sum(1)
            keep = np.sort(np.argsort(scores)[-cfg.max_pop:])
            pop = [pop[j] for j in keep]
            pop_meta = [pop_meta[j] for j in keep]
            payoff = payoff[keep, :][:, keep]
            n_old = int((keep < n_old).sum())

        if cfg.skim_when in ("after", "both"):
            pop, pop_meta, payoff, n_old = skim_population(
                pop, pop_meta, payoff, n_old, cfg.n_skim, cfg.skim_fraction, cfg.n_accepted,
            )

        t2 = time.time()
        n_removed_total = n_prev - n_old
        n_survived_new = len(pop) - n_old
        survived_counts = Counter(m["method"] for m in pop_meta[n_old:])
        survived_by_method = ", ".join(f"{k}={v}" for k, v in sorted(survived_counts.items()))
        print(f"Removed {n_removed_total} ({n_capped} via max_pop cap), New {n_survived_new} [{survived_by_method}], took {t1 - t0:.2f}s payoff, {t2 - t1:.2f}s skim")
        payoff_flat = payoff.sum(1)
        payout_flat = payout.sum(1)

        work_pop_dicts = [
            {"id": m["id"], "method": m["method"], "parents": m["parents"], "genome": genome}
            for genome, m in zip(pop, pop_meta)
        ]
        logger.log(
            {
                "gen": gen_offset + i,
                "t": t2,
                "payoff_s": round(t1 - t0, 4),
                "skim_s": round(t2 - t1, 4),
                "pop_size": len(pop),
                "n_added": len(new_individuals),
                "n_added_by_method": dict(method_counts),
                "n_removed": n_removed_total,
                "n_capped": n_capped,
                "n_survived_new": n_survived_new,
                "n_survived_new_by_method": dict(survived_counts),
                "payoff_mean": round(float(payoff_flat.mean()), 4),
                "payoff_std": round(float(payoff_flat.std()), 4),
                "payoff_min": int(payoff_flat.min()),
                "payoff_max": int(payoff_flat.max()),
                "payout_mean": round(float(payout_flat.mean()), 4),
                "payout_std": round(float(payout_flat.std()), 4),
                "payout_min": int(payout_flat.min()),
                "payout_max": int(payout_flat.max()),
            },
            work_pop=work_pop_dicts,
        )


if __name__ == "__main__":
    cfg = MainConfig()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    main(cfg)

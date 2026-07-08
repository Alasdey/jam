"""
Unified generation loop.

One run = repeated _step() over an explicit RunState:
  1. create   — seeded genomes (gen 0), fresh randoms, genetic offspring
                (bootstrap randoms while the population is still empty)
  2. evaluate — PayoffEngine.extend() grows the square self-play payoff matrix
  3. select   — each configured selection step maps payoff -> surviving indices

_step is a pure function of (state, rng): a future multi-pool/island driver
can hold several RunStates and migrate individuals between them without
touching this module.
"""

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from config import RunConfig, compat_key
from core.types import Individual, Program
from creation import build_creator
from creation.base import Creator
from creation.offspring import make_offspring
from interpreters import build_interpreter
from loggers.experiment_logger import ExperimentLogger
from loggers.run_io import load_checkpoint, reconstruct_population
from rewards.base import build_reward
from rewards.payoff import PayoffEngine
from selection.base import SelectFn, build_selection


@dataclass
class RunState:
    gen: int  # next generation to run
    next_id: int
    pop: list[Individual]
    payoff: np.ndarray  # square self-play matrix, rows/cols aligned with pop


@dataclass
class GenRecord:
    metrics: dict
    births: list[Individual]
    survivor_ids: list[int]
    scores: list[int]


def _stat_block(prefix: str, row_sums: np.ndarray) -> dict:
    if row_sums.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
                f"{prefix}_min": 0, f"{prefix}_max": 0}
    return {
        f"{prefix}_mean": round(float(row_sums.mean()), 4),
        f"{prefix}_std": round(float(row_sums.std()), 4),
        f"{prefix}_min": int(row_sums.min()),
        f"{prefix}_max": int(row_sums.max()),
    }


def _step(
    state: RunState,
    cfg: RunConfig,
    creator: Creator,
    interp,
    engine: PayoffEngine,
    selectors: Sequence[SelectFn],
    seed_genomes: Sequence[Program] = (),
) -> tuple[RunState, GenRecord]:
    t0 = time.time()
    old_pop = state.pop
    n_old = len(old_pop)
    next_id = state.next_id
    new_inds: list[Individual] = []

    for genome in seed_genomes:
        new_inds.append(Individual(next_id, list(genome), "seeded", [], state.gen))
        next_id += 1

    n_random = cfg.n_random + (cfg.n_init if state.gen == 0 and n_old == 0 else 0)
    for _ in range(n_random):
        new_inds.append(Individual(next_id, creator.random(), "random", [], state.gen))
        next_id += 1

    if n_old > 0:
        offspring, methods, parent_idx = make_offspring(
            creator=creator,
            survivors=[ind.genome for ind in old_pop],
            n_offspring=cfg.n_offspring,
            gc=cfg.experiment.genetics,
            interp=interp,
        )
        for genome, method, pidx in zip(offspring, methods, parent_idx):
            parents = [old_pop[i].id for i in pidx]
            new_inds.append(Individual(next_id, genome, method, parents, state.gen))
            next_id += 1
    else:
        # bootstrap: nothing to breed from yet (seeded genomes only join the
        # breeding pool once they survive into the next generation)
        for _ in range(cfg.n_offspring):
            new_inds.append(Individual(next_id, creator.random(), "random", [], state.gen))
            next_id += 1

    pop = old_pop + new_inds
    t_create = time.time()

    payoff = engine.extend(
        state.payoff,
        [ind.genome for ind in old_pop],
        [ind.genome for ind in new_inds],
    )
    t_payoff = time.time()
    payout_sums = payoff.sum(axis=1)  # pre-selection ground truth

    n_removed_by_step: list[int] = []
    for select in selectors:
        idx = select(payoff)
        n_removed_by_step.append(int(len(pop) - len(idx)))
        pop = [pop[i] for i in idx.tolist()]
        payoff = payoff[np.ix_(idx, idx)]
    t_select = time.time()

    new_ids = {ind.id for ind in new_inds}
    new_surviving = [ind for ind in pop if ind.id in new_ids]
    births = list(new_inds) if cfg.log_all_births else new_surviving
    survivor_ids = [ind.id for ind in pop]
    scores = [int(s) for s in payoff.sum(axis=1)]

    metrics = {
        "gen": state.gen,
        "t": t_select,
        "create_s": round(t_create - t0, 4),
        "payoff_s": round(t_payoff - t_create, 4),
        "select_s": round(t_select - t_payoff, 4),
        "pop_size": len(pop),
        "n_added": len(new_inds),
        "n_added_by_method": dict(Counter(ind.method for ind in new_inds)),
        "n_removed": (n_old + len(new_inds)) - len(pop),
        "n_removed_by_step": n_removed_by_step,
        "n_survived_new": len(new_surviving),
        "n_survived_new_by_method": dict(Counter(ind.method for ind in new_surviving)),
        **_stat_block("payoff", payoff.sum(axis=1)),
        **_stat_block("payout", payout_sums),
    }

    new_state = RunState(state.gen + 1, next_id, pop, payoff)
    return new_state, GenRecord(metrics, births, survivor_ids, scores)


def _resume_state(cfg: RunConfig) -> RunState:
    with open(Path(cfg.out_dir) / "config.json") as f:
        saved = json.load(f)
    saved_key = compat_key(saved["experiment"])
    current_key = compat_key(asdict(cfg.experiment))
    if saved_key != current_key:
        raise ValueError(
            f"Interpreter compat_key mismatch: run dir was produced with {saved_key}, "
            f"current config gives {current_key}"
        )

    checkpoint = load_checkpoint(cfg.out_dir)
    records = reconstruct_population(cfg.out_dir)
    record_ids = [r["id"] for r in records]
    if record_ids != checkpoint["pop_ids"]:
        raise ValueError("Checkpoint pop_ids do not match the reconstructed survivors")

    pop = [
        Individual(r["id"], r["genome"], r["method"], r["parents"], r["born_gen"])
        for r in records
    ]
    random.setstate(checkpoint["py_random_state"])
    np.random.set_state(checkpoint["np_random_state"])
    return RunState(checkpoint["gen"], checkpoint["next_id"], pop, checkpoint["payoff"])


def run(cfg: RunConfig) -> None:
    resuming = bool(cfg.resume_from)

    seed_genomes: list[Program] = []
    meta_extra: dict = {}
    if not resuming:
        if cfg.seed is None:
            cfg.seed = int.from_bytes(os.urandom(4), "little")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        if cfg.seed_populations:
            from store.population_store import load_seed_genomes

            seed_genomes, provenance = load_seed_genomes(cfg.store_dir, cfg.seed_populations)
            meta_extra["seeded_provenance"] = provenance

    creator = build_creator(cfg.experiment)
    interp = build_interpreter(cfg.experiment)
    reward = build_reward(cfg.experiment)
    selectors = build_selection(cfg.selection)

    if resuming:
        state = _resume_state(cfg)
    else:
        state = RunState(gen=0, next_id=0, pop=[], payoff=np.zeros((0, 0), dtype=int))

    logger = ExperimentLogger(cfg.out_dir, cfg, **meta_extra)
    end_gen = state.gen + cfg.n_iter

    with PayoffEngine(cfg.experiment, reward) as engine:
        while state.gen < end_gen:
            gen = state.gen
            genomes = seed_genomes if gen == 0 else ()
            state, rec = _step(state, cfg, creator, interp, engine, selectors, genomes)

            save_payoff = None
            if cfg.payoff_every > 0 and (gen % cfg.payoff_every == 0 or gen == end_gen - 1):
                save_payoff = state.payoff
            logger.log_generation(
                gen, rec.metrics, rec.births, rec.survivor_ids, rec.scores, save_payoff
            )
            logger.write_checkpoint({
                "format_version": 1,
                "gen": state.gen,
                "next_id": state.next_id,
                "pop_ids": rec.survivor_ids,
                "payoff": state.payoff,
                "py_random_state": random.getstate(),
                "np_random_state": np.random.get_state(),
            })

            m = rec.metrics
            print(
                f"gen {gen:>6} | pop {m['pop_size']:>5} | "
                f"+{m['n_added']} -{m['n_removed']} "
                f"[new {m['n_survived_new']}] | "
                f"payoff {m['payoff_s']:.2f}s select {m['select_s']:.2f}s"
            )

    if cfg.publish_label:
        from store.population_store import publish

        pop_id = publish(cfg.store_dir, cfg.publish_label, cfg.out_dir)
        print(f"Published final population to store as {pop_id}")

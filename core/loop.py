"""
Unified generation loop, on the GPU.

One run = repeated _step() over an explicit RunState:
  1. create   — seeded genomes (gen 0), fresh randoms, genetic offspring
                (bootstrap randoms while the population is still empty)
  2. pre-select — RunConfig.pre_selection, on genomes alone (payoff=None):
                thins the prospective population before any matchup is spent on
                it. `dedupe` lives here — duplicates cost nothing to discover.
  3. evaluate — PayoffEngine.extend() grows the square self-play payoff matrix
  4. select   — RunConfig.selection: each step maps (payoff, population, rng)
                -> surviving indices

The population and its payoff matrix stay on the device for the whole run;
only the records to log (surviving newborns, survivor ids and scores, the
matrix every payoff_every generations) come back to the host.

Randomness comes from DeviceRng(seed, gen) (core/rng.py): a pure function of
the seed and the generation, so a resumed run draws exactly what the
uninterrupted one did and checkpoints hold no generator state.

_step is a pure function of (state, rng): a future multi-pool/island driver
can hold several RunStates and migrate individuals between them without
touching this module.
"""

import json
import operator
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from config import RunConfig
from core.config_keys import compat_key
from core.device import cupy
from core.population import METHODS, Population
from core.rng import DeviceRng
from core.types import Program
from creation import build_creator
from loggers.experiment_logger import ExperimentLogger
from loggers.rewind import rewind_run
from loggers.run_io import last_gen, load_checkpoint, max_birth_id, reconstruct_population
from rewards.base import build_reward
from rewards.payoff import PayoffEngine
from selection.base import SelectFn, build_selection


@dataclass
class RunState:
    gen: int  # next generation to run
    next_id: int
    pop: Population
    payoff: object  # device int32 square self-play matrix, rows/cols aligned with pop


@dataclass
class GenRecord:
    metrics: dict
    births: Population  # newborns to persist
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


def _by_method(codes: np.ndarray) -> dict:
    counts = np.bincount(codes, minlength=len(METHODS))
    return {METHODS[m]: int(c) for m, c in enumerate(counts) if c}


def _now() -> float:
    cupy().cuda.Device().synchronize()  # stage timings count the device work
    return time.time()


def _step(
    state: RunState,
    cfg: RunConfig,
    creator,
    engine: PayoffEngine,
    selectors: Sequence[SelectFn],
    pre_selectors: Sequence[SelectFn] = (),
    seeded: Optional[Population] = None,
    rng: Optional[DeviceRng] = None,
) -> tuple[RunState, GenRecord]:
    cp = cupy()
    rng = rng if rng is not None else DeviceRng(cfg.seed, state.gen)
    t0 = _now()
    old_pop, old_payoff = state.pop, state.payoff
    n_old = len(old_pop)
    next_id = state.next_id

    parts = []
    if seeded is not None and len(seeded):
        seeded.ids = np.arange(next_id, next_id + len(seeded), dtype=np.int64)
        seeded.born_gen[:] = state.gen
        next_id += len(seeded)
        parts.append(seeded)
    n_random = cfg.n_random + (cfg.n_init if state.gen == 0 and n_old == 0 else 0)
    # Without survivors, the offspring slots are filled with random trees: nothing
    # to breed from yet (seeded genomes join the breeding pool next generation).
    created = creator.create(old_pop, n_random, cfg.n_offspring, rng, state.gen, next_id)
    next_id += len(created)
    parts.append(created)
    new = Population.concat(parts)

    # Pre-evaluation selection, on genomes alone: thins the prospective
    # population before any matchup is spent on it. Survivors stay sorted, so
    # the old block keeps its order and the payoff is simply reindexed.
    n_removed_by_pre_step: list[int] = []
    if pre_selectors:
        prospective = Population.concat([old_pop, new])
        alive = np.arange(len(prospective))
        current = prospective
        for select in pre_selectors:
            idx = cp.asnumpy(cp.asarray(select(None, current, rng))).astype(np.int64)
            n_removed_by_pre_step.append(int(len(alive) - len(idx)))
            alive = alive[idx]
            current = current.take(idx)
        old_keep = alive[alive < n_old]
        new_keep = alive[alive >= n_old] - n_old
        if len(old_keep) < n_old:
            old_pop = old_pop.take(old_keep)
            old_payoff = old_payoff[cp.ix_(cp.asarray(old_keep), cp.asarray(old_keep))]
        new = new.take(new_keep)

    pop = Population.concat([old_pop, new])
    t_create = _now()

    payoff = engine.extend(old_payoff, pop, len(old_pop))
    t_payoff = _now()
    payout_sums = cp.asnumpy(payoff.sum(axis=1, dtype=cp.int64))  # ground truth, before selection thins it

    n_removed_by_step: list[int] = []
    for select in selectors:
        idx = cp.asarray(select(payoff, pop, rng)).astype(cp.int64)
        n_removed_by_step.append(int(len(pop) - len(idx)))
        pop = pop.take(idx)
        payoff = payoff[cp.ix_(idx, idx)]
    t_select = _now()

    new_ids = set(new.ids.tolist())
    surviving_new = np.array([i for i, pid in enumerate(pop.ids.tolist()) if pid in new_ids], dtype=np.int64)
    births = new if cfg.log_all_births else pop.take(surviving_new)
    scores = cp.asnumpy(payoff.sum(axis=1, dtype=cp.int64))

    metrics = {
        "gen": state.gen,
        "t": t_select,
        "create_s": round(t_create - t0, 4),
        "payoff_s": round(t_payoff - t_create, 4),
        "select_s": round(t_select - t_payoff, 4),
        "pop_size": len(pop),
        "n_added": len(new),
        "n_added_by_method": _by_method(new.method),
        "n_removed_pre": sum(n_removed_by_pre_step),
        "n_removed_by_pre_step": n_removed_by_pre_step,
        "n_removed": (len(old_pop) + len(new)) - len(pop),
        "n_removed_by_step": n_removed_by_step,
        "n_survived_new": len(surviving_new),
        "n_survived_new_by_method": _by_method(pop.method[surviving_new]),
        **_stat_block("payoff", scores),
        **_stat_block("payout", payout_sums),
    }

    new_state = RunState(state.gen + 1, next_id, pop, payoff)
    return new_state, GenRecord(metrics, births, pop.ids.tolist(), [int(s) for s in scores])


def _resume_state(cfg: RunConfig, engine: PayoffEngine) -> RunState:
    with open(Path(cfg.out_dir) / "config.json") as f:
        saved = json.load(f)
    saved_key = compat_key(saved["experiment"])
    current_key = compat_key(asdict(cfg.experiment))
    if saved_key != current_key:
        raise ValueError(
            f"Interpreter compat_key mismatch: run dir was produced with {saved_key}, "
            f"current config gives {current_key}"
        )
    # The persisted seed drives every generation's draws (DeviceRng(seed, gen)).
    cfg.seed = saved["seed"]

    if cfg.resume_gen is not None:
        return _resume_generation_state(cfg, saved, engine)

    checkpoint = load_checkpoint(cfg.out_dir)
    _trim_metrics(Path(cfg.out_dir), checkpoint["gen"])
    records = reconstruct_population(cfg.out_dir)
    if [r["id"] for r in records] != checkpoint["pop_ids"]:
        raise ValueError("Checkpoint pop_ids do not match the reconstructed survivors")
    payoff = cupy().asarray(np.asarray(checkpoint["payoff"], dtype=np.int32))
    return RunState(checkpoint["gen"], checkpoint["next_id"], _population(records), payoff)


def _trim_metrics(root: Path, gen: int) -> None:
    """Drop metrics of generations >= gen: a run stopped between logging a generation and
    checkpointing it runs that generation again on resume (its other files are rewritten)."""
    path = root / "metrics.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    kept = [line for line in lines if json.loads(line)["gen"] < gen]
    if len(kept) != len(lines):
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(kept))
        os.replace(tmp, path)


def _resume_generation_state(cfg: RunConfig, saved: dict, engine: PayoffEngine) -> RunState:
    """Rebuild a historical population; later generations draw from DeviceRng(seed, gen) as usual."""
    root = Path(cfg.out_dir)
    records = reconstruct_population(cfg.out_dir, cfg.resume_gen)
    ids = [r["id"] for r in records]
    pop = _population(records)

    # Preserve the ID high-water mark, including births discarded by rollback.
    next_id = max_birth_id(cfg.out_dir, last_gen(cfg.out_dir)) + 1
    if (root / "checkpoint.pkl").exists():
        next_id = max(next_id, load_checkpoint(cfg.out_dir)["next_id"])

    # Old snapshots have no embedded reward/config provenance. Only reuse one
    # when every recorded invocation agrees on payoff semantics.
    configs = [saved]
    for path in root.glob("config_resumed_*.json"):
        with open(path) as f:
            configs.append(json.load(f))
    current_key = compat_key(asdict(cfg.experiment))
    reusable = all(
        c["experiment"]["reward"] == cfg.experiment.reward
        and compat_key(c["experiment"]) == current_key for c in configs
    )
    path = root / "payoffs" / f"payoff_{cfg.resume_gen:06d}.npz"
    if path.exists() and reusable:
        with np.load(path, allow_pickle=False) as data:
            payoff = data["payoff"]
            if data["ids"].tolist() != ids:
                raise ValueError("Historical payoff IDs do not match survivors")
        if payoff.shape != (len(pop), len(pop)) or not np.isfinite(payoff).all():
            raise ValueError("Historical payoff must be a finite square survivor matrix")
        payoff = cupy().asarray(payoff.astype(np.int32))
    else:
        payoff = engine.square(pop)
    return RunState(cfg.resume_gen + 1, next_id, pop, payoff)


def _population(records: list[dict]) -> Population:
    if not records:
        return Population.empty()
    return Population.from_records(records)


def _checkpoint(state: RunState) -> dict:
    return {
        "format_version": 2,
        "gen": state.gen,
        "next_id": state.next_id,
        "pop_ids": state.pop.ids.tolist(),
        "payoff": cupy().asnumpy(state.payoff),
    }


def _seeded(genomes: Sequence[Program]) -> Optional[Population]:
    if not genomes:
        return None
    n = len(genomes)
    return Population.from_genomes(genomes, np.zeros(n, dtype=np.int64), "seeded", [[] for _ in range(n)], 0)


def run(cfg: RunConfig) -> None:
    resuming = bool(cfg.resume_from)
    if cfg.resume_gen is not None:
        if not resuming:
            raise ValueError("resume_gen requires resume_from")
        if isinstance(cfg.resume_gen, bool):
            raise ValueError("resume_gen must be a non-negative integer")
        try:
            cfg.resume_gen = operator.index(cfg.resume_gen)
        except TypeError as exc:
            raise ValueError("resume_gen must be a non-negative integer") from exc
        if cfg.resume_gen < 0:
            raise ValueError("resume_gen must be a non-negative integer")

    seed_genomes: list[Program] = []
    meta_extra: dict = {}
    if not resuming:
        if cfg.seed is None:
            cfg.seed = int.from_bytes(os.urandom(4), "little")
        if cfg.seed_populations:
            from store.population_store import load_seed_genomes

            seed_genomes, provenance = load_seed_genomes(cfg.store_dir, cfg.seed_populations)
            meta_extra["seeded_provenance"] = provenance

    reward = build_reward(cfg.experiment)
    pre_selectors = build_selection(cfg.pre_selection, pre=True)
    selectors = build_selection(cfg.selection)

    with PayoffEngine(cfg.experiment, reward) as engine:
        creator = build_creator(cfg.experiment, engine.runtime)
        if resuming:
            state = _resume_state(cfg, engine)
        else:
            state = RunState(gen=0, next_id=0, pop=Population.empty(),
                             payoff=cupy().zeros((0, 0), dtype=cupy().int32))

        if cfg.resume_gen is not None:
            backup = rewind_run(cfg.out_dir, cfg.resume_gen, _checkpoint(state))
            meta_extra["rollback_backup"] = str(backup)
            print(f"Restored generation {cfg.resume_gen}; backup: {backup}")

        logger = ExperimentLogger(cfg.out_dir, cfg, **meta_extra)
        end_gen = state.gen + cfg.n_iter

        while state.gen < end_gen:
            gen = state.gen
            seeded = _seeded(seed_genomes) if gen == 0 else None
            state, rec = _step(state, cfg, creator, engine, selectors, pre_selectors, seeded)

            save_payoff = None
            if cfg.payoff_every > 0 and (gen % cfg.payoff_every == 0 or gen == end_gen - 1):
                save_payoff = cupy().asnumpy(state.payoff)
            logger.log_generation(
                gen, rec.metrics, rec.births, rec.survivor_ids, rec.scores, save_payoff
            )
            logger.write_checkpoint(_checkpoint(state))

            m = rec.metrics
            print(
                f"gen {gen:>6} | pop {m['pop_size']:>5} | "
                f"+{m['n_added']} -{m['n_removed']} "
                f"[new {m['n_survived_new']}] | "
                f"create {m['create_s']:.2f}s payoff {m['payoff_s']:.2f}s select {m['select_s']:.2f}s"
            )

    if cfg.publish_label:
        from store.population_store import publish

        pop_id = publish(cfg.store_dir, cfg.publish_label, cfg.out_dir)
        print(f"Published final population to store as {pop_id}")

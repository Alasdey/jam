"""
The online loop: a continuous stream of small batches of new codes, each
created, tested against the population, selected and inserted. Everything
runs on the GPU; there are no generations.

  create batch t — a worker process (core/workers.py, core/batch.py) makes
                   n_random + n_offspring codes from a snapshot of the
                   population and drops those RunConfig.pre_selection removes
                   (they never cost a matchup); the coordinator gets them at once.
  test batch t   — the same worker tests them against the snapshot, the new
                   codes of the batches created since the snapshot (whichever
                   of them are members when batch t is inserted), and each other.
  insert batch t — the coordinator (this process), strictly in batch order:
                   runs pre_selection again against the current population
                   (duplicates of members that joined after the snapshot) and
                   applies RunConfig.selection to the square payoff matrix of
                   members and new codes. Every entry is already computed, so
                   inserting costs no matchup.

Several batches are in flight, so the workers keep the GPU busy while the
coordinator inserts. Batch t is bred from the population as it stood after
batch t - in_flight was inserted (or at the start of its epoch): a fixed lag,
not whatever was there when a worker got free, so a run is a pure function of
its seed and config, whatever the number of workers or their timing.

Epochs: every epoch_batches insertions the stream drains (no batch of the
next epoch is bred before the last of this one is inserted), then the epoch
is logged and checkpointed in the run-directory format of docs/formats.md,
where "gen" counts epochs: births of the epoch, survivors, the payoff matrix
every payoff_every epochs. A resumed run continues exactly as the
uninterrupted run did.

Randomness: DeviceRng(seed, t) makes batch t (in its worker) and
DeviceRng(seed, t, first_call=SELECTION_CALLS) inserts it (here); the
initial population draws as batch INIT_BATCH.
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
from core.batch import build_batch_maker
from core.config_keys import compat_key
from core.device import cupy
from core.population import METHODS, Population
from core.rng import DeviceRng
from core.types import Program
from creation import build_creator
from loggers.experiment_logger import ExperimentLogger
from loggers.rewind import rewind_run
from loggers.run_io import last_gen, load_checkpoint, load_survivors, max_birth_id, reconstruct_population
from rewards.base import build_reward
from rewards.payoff import PayoffEngine
from selection.base import SelectFn, build_selection

INIT_BATCH = 2**32 - 1  # the Philox "gen" of the initial population's draws
SELECTION_CALLS = 1 << 31  # the coordinator's first Philox call for a batch (workers start at 0)
CHECKPOINT_FORMAT = 3  # 2 was the generational loop's


@dataclass
class RunState:
    epoch: int       # next epoch to run
    next_batch: int  # index of the next batch to make
    next_id: int     # first id of that batch; batch t takes ids next_id + (t - next_batch) * batch size
    pop: Population
    payoff: object   # device int32 square self-play matrix, rows/cols aligned with pop


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


def _sync() -> None:
    cupy().cuda.Device().synchronize()


def _pre_select(pop: Population, new: Population, pre_selectors: Sequence[SelectFn], rng) -> tuple[np.ndarray, list]:
    """Indices of the new codes pre_selection keeps next to pop, and how many new codes each step dropped."""
    cp = cupy()
    n_p = len(pop)
    current = Population.concat([pop, new])
    alive = np.arange(len(current))
    removed = []
    for select in pre_selectors:
        idx = cp.asnumpy(cp.asarray(select(None, current, rng))).astype(np.int64)
        kept = alive[idx]
        if np.count_nonzero(kept < n_p) != n_p:
            raise RuntimeError("a pre_selection step dropped a member of the population; "
                               "pre_selection may only filter new codes")
        removed.append(int(np.count_nonzero(alive >= n_p) - np.count_nonzero(kept >= n_p)))
        alive = kept
        current = current.take(idx)
    return alive[alive >= n_p] - n_p, removed


def _select(payoff, pop: Population, selectors: Sequence[SelectFn], rng) -> tuple[np.ndarray, list]:
    cp = cupy()
    alive = np.arange(len(pop))
    removed = []
    for select in selectors:
        idx = cp.asnumpy(cp.asarray(select(payoff, pop, rng))).astype(np.int64)
        removed.append(int(len(pop) - len(idx)))
        alive = alive[idx]
        pop = pop.take(idx)
        payoff = payoff[cp.ix_(cp.asarray(idx), cp.asarray(idx))]
    return alive, removed


class Coordinator:
    """Inserts batches into the population, in batch order; holds the population and its matrix."""

    def __init__(self, cfg: RunConfig, engine: PayoffEngine, pre_selectors, selectors, logger: ExperimentLogger):
        self.cfg = cfg
        self.engine = engine
        self.pre_selectors = pre_selectors
        self.selectors = selectors
        self.logger = logger

    def insert(self, pop: Population, payoff, created: dict, tested: dict, opponent_ids: np.ndarray, epoch: int,
               wid) -> tuple:
        """
        (population, matrix, births to log) after inserting one batch; logs its
        metrics. `created` and `tested` are the batch's results (core/batch.py),
        opponent_ids the ids its new codes were tested against, in order:
        every member of pop is among them.
        """
        cp = cupy()
        t = created["t"]
        rng = DeviceRng(self.cfg.seed, t, first_call=SELECTION_CALLS)
        t0 = time.perf_counter()
        new = Population.from_host(created["new"])
        n_p, n_o = len(pop), len(opponent_ids)

        removed_pre = list(created["n_removed_by_pre_step"])
        keep = np.arange(len(new))
        if self.pre_selectors and len(new):
            keep, removed = _pre_select(pop, new, self.pre_selectors, rng)
            removed_pre = [a + b for a, b in zip(removed_pre, removed)]
            new = new.take(keep)
        k = len(new)

        position = {pid: i for i, pid in enumerate(opponent_ids.tolist())}
        try:
            members = cp.asarray(np.array([position[pid] for pid in pop.ids.tolist()], dtype=np.int64))
        except KeyError as exc:
            raise RuntimeError(f"member {exc} of the population is not among batch {t}'s opponents") from None
        matrix = cp.zeros((n_p + k, n_p + k), dtype=cp.int32)
        matrix[:n_p, :n_p] = payoff
        if k:
            rows = cp.asarray(tested["rows"][keep])     # new x opponents + new
            cols = cp.asarray(tested["cols"][:, keep])  # opponents x new
            if n_p:
                matrix[n_p:, :n_p] = rows[:, members]
                matrix[:n_p, n_p:] = cols[members, :]
            matrix[n_p:, n_p:] = rows[:, cp.asarray(n_o + keep)]
        combined = Population.concat([pop, new])
        payout_sums = cp.asnumpy(matrix.sum(axis=1, dtype=cp.int64))  # before selection thins it

        idx, removed_by_step = _select(matrix, combined, self.selectors, rng)
        pop = combined.take(idx)
        sel = cp.asarray(idx)
        payoff = matrix[cp.ix_(sel, sel)]
        surviving_new = np.flatnonzero(idx >= n_p)
        births = new if self.cfg.log_all_births else pop.take(surviving_new)
        scores = cp.asnumpy(payoff.sum(axis=1, dtype=cp.int64))
        _sync()

        self.logger.log_metrics({
            "gen": epoch,
            "batch": t,
            "t": time.time(),
            "worker": wid,
            "create_s": round(created["create_s"], 4),
            "test_s": round(tested["test_s"], 4),
            "insert_s": round(time.perf_counter() - t0, 4),
            "oom_retries": created["oom_retries"] + tested["oom_retries"],
            "pop_size": len(pop),
            "n_created": created["n_created"],
            "n_added": k,
            "n_added_by_method": _by_method(new.method),
            "n_removed_pre": sum(removed_pre),
            "n_removed_by_pre_step": removed_pre,
            "n_opponents": n_o,
            "n_matchups": tested["n_matchups"],
            "n_removed": n_p + k - len(pop),
            "n_removed_by_step": removed_by_step,
            "n_survived_new": len(surviving_new),
            "n_survived_new_by_method": _by_method(pop.method[surviving_new]),
            **_stat_block("payoff", scores),
            **_stat_block("payout", payout_sums),
        })
        return pop, payoff, births


def _batch_size(cfg: RunConfig) -> int:
    return cfg.n_random + cfg.n_offspring


def _run_epoch(state: RunState, cfg: RunConfig, coord: Coordinator, maker, pool, births_parts: list,
               end_epoch: int) -> RunState:
    """Stream epoch_batches batches through the population, then log and checkpoint the epoch."""
    cp = cupy()
    n_batches, lag, size = cfg.epoch_batches, cfg.stream.in_flight, _batch_size(cfg)
    t0 = state.next_batch
    t_end = t0 + n_batches
    pop, payoff = state.pop, state.payoff

    def source(t: int) -> int:  # the insertion whose population batch t is bred from
        return max(t - lag, t0 - 1)

    def between(t: int) -> range:  # batches whose new codes batch t is tested against, besides its snapshot
        return range(source(t) + 1, t)

    snapshots = {t0 - 1: (pop.to_host(), pop.ids.copy())}
    created: dict[int, dict] = {}
    tested: dict[int, dict] = {}
    owner: dict[int, Optional[int]] = {}
    test_sent: set[int] = set()

    def create_job(t: int) -> dict:
        return {"kind": "create", "t": t, "snapshot": snapshots[source(t)][0],
                "first_id": state.next_id + (t - t0) * size, "born_gen": state.epoch, "n_random": cfg.n_random}

    def test_job(t: int) -> dict:
        return {"kind": "test", "t": t, "opponents": [created[j]["new"] for j in between(t)]}

    next_create = next_insert = t0
    idle = list(range(cfg.stream.n_workers))
    wall0 = time.perf_counter()
    waited = inserting = 0.0
    n_tested = n_kept = 0
    while next_insert < t_end:
        if pool is None:  # one batch at a time, here, in the same order of dependencies
            t = next_insert
            owner[t] = None
            created[t] = maker.create(create_job(t))
            tested[t] = maker.test(test_job(t))
        else:
            while idle and next_create < t_end and source(next_create) < next_insert:
                owner[next_create] = idle.pop(0)
                pool.submit(owner[next_create], create_job(next_create))
                next_create += 1
            tw = time.perf_counter()
            wid, result = pool.result()
            waited += time.perf_counter() - tw
            if result["kind"] == "created":
                created[result["t"]] = result  # the worker keeps the batch until its test
            else:
                tested[result["t"]] = result
                idle.append(wid)
            for t in sorted(set(created) - test_sent):
                if all(j in created for j in between(t)):
                    pool.submit(owner[t], test_job(t))
                    test_sent.add(t)
        while next_insert in tested:
            t = next_insert
            opponent_ids = np.concatenate([snapshots[source(t)][1]] + [created[j]["new"]["ids"] for j in between(t)])
            ti = time.perf_counter()
            pop, payoff, births = coord.insert(pop, payoff, created[t], tested.pop(t), opponent_ids, state.epoch,
                                               owner.pop(t))
            inserting += time.perf_counter() - ti
            n_tested += len(created[t]["new"]["ids"])
            n_kept += len(births)
            births_parts.append(births)
            snapshots[t] = (pop.to_host(), pop.ids.copy())
            next_insert += 1
            oldest = source(min(next_insert, t_end - 1))  # no later batch needs anything older
            for s in [s for s in snapshots if s < oldest]:
                del snapshots[s]
            for j in [j for j in created if j <= oldest]:
                del created[j]
                test_sent.discard(j)

    wall = time.perf_counter() - wall0
    epoch = state.epoch
    scores = cp.asnumpy(payoff.sum(axis=1, dtype=cp.int64))
    save_payoff = None
    if cfg.payoff_every > 0 and (epoch % cfg.payoff_every == 0 or epoch == end_epoch - 1):
        save_payoff = cp.asnumpy(payoff)
    coord.logger.log_population(epoch, Population.concat(births_parts), pop.ids.tolist(), [int(s) for s in scores],
                                save_payoff, next_batch=t_end)
    births_parts.clear()
    new_state = RunState(epoch + 1, t_end, state.next_id + n_batches * size, pop, payoff)
    coord.logger.write_checkpoint(_checkpoint(new_state))
    print(f"epoch {epoch:>5} | batches {t0}-{t_end - 1} | pop {len(pop):>4} | tested {n_tested:>6} new codes, "
          f"{n_tested / wall:,.0f}/s | inserted {n_kept:>5} | {wall:.1f}s "
          f"(coordinator waited {waited:.1f}s, inserted {inserting:.1f}s)", flush=True)
    return new_state


def _initial_state(cfg: RunConfig, engine: PayoffEngine, creator, coord: Coordinator,
                   seed_genomes: Sequence[Program], births_parts: list) -> RunState:
    """Seeded genomes and n_init random trees, tested and selected: the population the stream starts from."""
    cp = cupy()
    parts = []
    n_seed = len(seed_genomes)
    if n_seed:
        parts.append(Population.from_genomes(seed_genomes, np.arange(n_seed), "seeded", [[] for _ in range(n_seed)], 0))
    if cfg.n_init:
        parts.append(creator.create(Population.empty(), cfg.n_init, 0, DeviceRng(cfg.seed, INIT_BATCH), 0, n_seed))
    pop = Population.concat(parts)
    n0 = len(pop)
    if not n0:
        return RunState(0, 0, 0, pop, cp.zeros((0, 0), dtype=cp.int32))
    rng = DeviceRng(cfg.seed, INIT_BATCH, first_call=SELECTION_CALLS)
    removed_pre = [0] * len(coord.pre_selectors)
    if coord.pre_selectors:
        keep, removed_pre = _pre_select(Population.empty(), pop, coord.pre_selectors, rng)
        pop = pop.take(keep)
    t0 = time.perf_counter()
    matrix = engine.square(pop)
    test_s = time.perf_counter() - t0
    payout_sums = cp.asnumpy(matrix.sum(axis=1, dtype=cp.int64))
    idx, removed_by_step = _select(matrix, pop, coord.selectors, rng)
    births = pop if cfg.log_all_births else pop.take(idx)
    n_added = len(pop)
    added_by_method = _by_method(pop.method)
    pop = pop.take(idx)
    sel = cp.asarray(idx)
    payoff = matrix[cp.ix_(sel, sel)]
    births_parts.append(births)
    scores = cp.asnumpy(payoff.sum(axis=1, dtype=cp.int64))
    coord.logger.log_metrics({
        "gen": 0, "batch": -1, "t": time.time(), "worker": None, "create_s": 0.0, "test_s": round(test_s, 4),
        "insert_s": 0.0, "oom_retries": 0, "pop_size": len(pop), "n_created": n0, "n_added": n_added,
        "n_added_by_method": added_by_method, "n_removed_pre": sum(removed_pre),
        "n_removed_by_pre_step": removed_pre, "n_matchups": n_added * n_added,
        "n_removed": n_added - len(pop), "n_removed_by_step": removed_by_step,
        "n_survived_new": len(pop), "n_survived_new_by_method": _by_method(pop.method),
        **_stat_block("payoff", scores), **_stat_block("payout", payout_sums),
    })
    return RunState(0, 0, n0, pop, payoff)


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
    # The persisted seed drives every batch's draws (DeviceRng(seed, t)).
    cfg.seed = saved["seed"]

    if cfg.resume_gen is not None:
        return _resume_generation_state(cfg, saved, engine)

    checkpoint = load_checkpoint(cfg.out_dir)
    _check_online(checkpoint.get("format_version"), cfg.out_dir)
    _trim_metrics(Path(cfg.out_dir), checkpoint["gen"])
    records = reconstruct_population(cfg.out_dir)
    if [r["id"] for r in records] != checkpoint["pop_ids"]:
        raise ValueError("Checkpoint pop_ids do not match the reconstructed survivors")
    payoff = cupy().asarray(np.asarray(checkpoint["payoff"], dtype=np.int32))
    return RunState(checkpoint["gen"], checkpoint["next_batch"], checkpoint["next_id"], _population(records), payoff)


def _check_online(format_version, run_dir) -> None:
    if format_version != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{run_dir} was made by the generational loop (checkpoint format {format_version}), which this "
            "online loop cannot continue; publish its population (store.publish) and seed a new run from it "
            "with RunConfig.seed_populations"
        )


def _trim_metrics(root: Path, gen: int) -> None:
    """Drop metrics of epochs >= gen: a run stopped inside an epoch runs that epoch again on resume."""
    path = root / "metrics.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    kept = [line for line in lines if json.loads(line)["gen"] < gen]
    if len(kept) != len(lines):
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(kept))
        os.replace(tmp, path)


def _resume_generation_state(cfg: RunConfig, saved: dict, engine: PayoffEngine) -> RunState:
    """Rebuild the population after an earlier epoch; the stream continues from there."""
    root = Path(cfg.out_dir)
    survivors = load_survivors(cfg.out_dir, cfg.resume_gen)
    if "next_batch" not in survivors:
        _check_online(None, cfg.out_dir)
    records = reconstruct_population(cfg.out_dir, cfg.resume_gen)
    ids = [r["id"] for r in records]
    pop = _population(records)

    # Ids above every id the run handed out, including births discarded by the rewind.
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
    return RunState(cfg.resume_gen + 1, survivors["next_batch"], next_id, pop, payoff)


def _population(records: list[dict]) -> Population:
    if not records:
        return Population.empty()
    return Population.from_records(records)


def _checkpoint(state: RunState) -> dict:
    return {
        "format_version": CHECKPOINT_FORMAT,
        "gen": state.epoch,
        "next_batch": state.next_batch,
        "next_id": state.next_id,
        "pop_ids": state.pop.ids.tolist(),
        "payoff": cupy().asnumpy(state.payoff),
    }


def _validate(cfg: RunConfig) -> None:
    s = cfg.stream
    if cfg.n_random < 0 or cfg.n_offspring < 0 or _batch_size(cfg) < 1:
        raise ValueError("a batch needs at least one new code (n_random + n_offspring >= 1)")
    if cfg.epoch_batches < 1:
        raise ValueError("epoch_batches must be at least 1")
    if s.n_workers < 0 or s.in_flight < 1:
        raise ValueError("stream.n_workers must be >= 0 and stream.in_flight >= 1")
    if s.worker_memory_mb < 1 or s.coordinator_memory_mb < 1:
        raise ValueError("stream memory caps must be positive")


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
    _validate(cfg)

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

    pool_memory = cupy().get_default_memory_pool()
    previous_limit = pool_memory.get_limit()
    pool_memory.set_limit(size=cfg.stream.coordinator_memory_mb * 2**20)
    workers = None
    try:
        with PayoffEngine(cfg.experiment, reward) as engine:
            creator = build_creator(cfg.experiment, engine.runtime)
            if resuming:
                state = _resume_state(cfg, engine)
                if cfg.resume_gen is not None:
                    backup = rewind_run(cfg.out_dir, cfg.resume_gen, _checkpoint(state))
                    meta_extra["rollback_backup"] = str(backup)
                    print(f"Restored epoch {cfg.resume_gen}; backup: {backup}")
            logger = ExperimentLogger(cfg.out_dir, cfg, **meta_extra)
            coord = Coordinator(cfg, engine, pre_selectors, selectors, logger)
            births_parts: list = []
            if not resuming:
                if cfg.n_iter <= 0:
                    return
                state = _initial_state(cfg, engine, creator, coord, seed_genomes, births_parts)
            end_epoch = state.epoch + cfg.n_iter

            maker = build_batch_maker(cfg, engine) if cfg.stream.n_workers == 0 else None
            if cfg.stream.n_workers:
                from core.workers import WorkerPool

                workers = WorkerPool(cfg)
            while state.epoch < end_epoch:
                state = _run_epoch(state, cfg, coord, maker, workers, births_parts, end_epoch)
    finally:
        if workers is not None:
            workers.close()
        pool_memory.set_limit(size=previous_limit)

    if cfg.publish_label:
        from store.population_store import publish

        pop_id = publish(cfg.store_dir, cfg.publish_label, cfg.out_dir)
        print(f"Published final population to store as {pop_id}")

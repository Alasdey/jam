"""
One batch of the online loop (core/loop.py), in two steps on one process:

  create(job)  new codes from a snapshot of the population, filtered by
               RunConfig.pre_selection (against the snapshot): published to
               the coordinator at once, so later batches can test against them.
  test(job)    the new codes against their opponents: the snapshot, the new
               codes of the batches created since it (any of them may be a
               member by the time this batch is inserted), and each other.

A worker process makes batches (core/workers.py), or the coordinator itself
when StreamConfig.n_workers is 0. Either way a batch depends only on its jobs
and the run's seed: draws come from DeviceRng(seed, t).

Memory: a step that runs out of its process's CuPy pool cap is made again,
from the same draws, with halved launch budgets (payoff tiles and Treemo
launches), so the result is the same and the cap is never exceeded.
"""

import time

import numpy as np

from config import RunConfig
from core.device import cupy
from core.population import Population
from core.rng import DeviceRng


class BatchMaker:
    def __init__(self, cfg: RunConfig, engine, creator, pre_selectors):
        self.seed = cfg.seed
        self.n_offspring = cfg.n_offspring
        self.engine = engine
        self.creator = creator
        self.pre_selectors = pre_selectors
        self._held: dict[int, tuple] = {}  # batch -> (snapshot, new codes), between create and test

    def create(self, job: dict) -> dict:
        """
        job: "t" (batch index), "snapshot" (Population.to_host() of the
        population to breed from), "first_id", "born_gen", "n_random".
        """
        return self._retrying(self._create, job)

    def test(self, job: dict) -> dict:
        """
        job: "t", "opponents" (to_host() of the new codes of the batches
        between the snapshot and this one, in batch order). Returns "rows"
        [new, opponents] and "cols" [opponents, new], opponents being the
        snapshot, job["opponents"] and the new codes, in that order.
        """
        return self._retrying(self._test, job)

    def _retrying(self, step, job: dict) -> dict:
        cp = cupy()
        engine, runtime = self.engine, self.engine.runtime
        budgets = engine.batch, runtime.memory_mb
        retries = 0
        try:
            while True:
                try:
                    result = step(job)
                    result["oom_retries"] = retries
                    return result
                except cp.cuda.memory.OutOfMemoryError:
                    cp.get_default_memory_pool().free_all_blocks()
                    if engine.batch <= 1 and runtime.memory_mb <= 1:
                        raise
                    engine.batch = max(1, engine.batch // 2)
                    runtime.memory_mb = max(1, runtime.memory_mb // 2)
                    retries += 1
        finally:
            engine.batch, runtime.memory_mb = budgets

    def _create(self, job: dict) -> dict:
        cp = cupy()
        t0 = time.perf_counter()
        snap = Population.from_host(job["snapshot"])
        n_s = len(snap)
        rng = DeviceRng(self.seed, job["t"])
        kids = self.creator.create(snap, job["n_random"], self.n_offspring, rng, job["born_gen"], job["first_id"])
        n_created = len(kids)
        removed = [0] * len(self.pre_selectors)
        if self.pre_selectors and len(kids):
            current = Population.concat([snap, kids])
            alive = np.arange(len(current))
            for s, select in enumerate(self.pre_selectors):
                idx = cp.asnumpy(cp.asarray(select(None, current, rng))).astype(np.int64)
                kept = alive[idx]
                removed[s] = int(np.count_nonzero(alive >= n_s) - np.count_nonzero(kept >= n_s))
                alive = kept
                current = current.take(idx)
            kids = kids.take(alive[alive >= n_s] - n_s)  # members stay: only new codes are filtered
        self._held[job["t"]] = (snap, kids)
        cp.cuda.get_current_stream().synchronize()
        return {"kind": "created", "t": job["t"], "new": kids.to_host(), "n_created": n_created,
                "n_removed_by_pre_step": removed, "create_s": time.perf_counter() - t0}

    def _test(self, job: dict) -> dict:
        cp = cupy()
        t0 = time.perf_counter()
        snap, kids = self._held[job["t"]]
        block = Population.concat([snap] + [Population.from_host(h) for h in job["opponents"]] + [kids])
        n_o, k = len(block) - len(kids), len(kids)
        payoff = self.engine.square(block, cp.zeros((n_o, n_o), dtype=cp.int32), n_o)
        rows = cp.asnumpy(payoff[n_o:])
        cols = cp.asnumpy(payoff[:n_o, n_o:])
        del self._held[job["t"]]
        return {"kind": "tested", "t": job["t"], "rows": rows, "cols": cols, "n_matchups": 2 * k * n_o + k * k,
                "test_s": time.perf_counter() - t0}


def build_batch_maker(cfg: RunConfig, engine=None) -> BatchMaker:
    """A BatchMaker with its own payoff engine and creator (one per process)."""
    from creation import build_creator
    from rewards.base import build_reward
    from rewards.payoff import PayoffEngine
    from selection.base import build_selection

    if engine is None:
        engine = PayoffEngine(cfg.experiment, build_reward(cfg.experiment))
    creator = build_creator(cfg.experiment, engine.runtime)
    return BatchMaker(cfg, engine, creator, build_selection(cfg.pre_selection, pre=True))

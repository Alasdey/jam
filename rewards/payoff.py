"""
PayoffEngine: the one place matchups are scored, for the evolution loop, the
tournaments and the analysis tools. Everything runs on the GPU.

A reward scores a batch of matchups (core/matchups.py). The engine cuts a
payoff matrix into batches of about PayoffConfig.gpu_batch_matchups
matchups. In a square self-play matrix, tile (I, J) travels with its
transpose (J, I) for rewards that are not zero-sum, so both orientations of a
pair share their executions; zero-sum rewards compute one orientation and
derive the other as -A.T.
"""

from math import isqrt
from typing import Sequence

import numpy as np

from config import ExperimentConfig
from core.device import cupy
from core.matchups import Matchups
from core.population import Population
from interpreters.treemo_gpu.rules import extract_rules
from interpreters import build_runtime
from rewards.base import RewardSpec

_POOL_SLACK = 2**30  # cached-but-unused device memory released between batches


class PayoffEngine:
    def __init__(self, exp_cfg: ExperimentConfig, reward: RewardSpec):
        if exp_cfg.payoff.gpu_batch_matchups < 1:
            raise ValueError("payoff.gpu_batch_matchups must be at least 1")
        self.exp_cfg = exp_cfg
        self.reward = reward
        self.runtime = build_runtime(exp_cfg)
        self.batch = exp_cfg.payoff.gpu_batch_matchups

    def _score(self, pop: Population, rules, a, b):
        cp = cupy()
        values = cp.asarray(self.reward.fn(Matchups(self.runtime, pop, a, b, rules)))
        if values.shape != (len(a),):
            raise ValueError(
                f"reward {self.reward.name!r} returned shape {values.shape}; it must return one value "
                f"per matchup ({len(a)},)"
            )
        pool = cp.get_default_memory_pool()
        if pool.total_bytes() - pool.used_bytes() > _POOL_SLACK:
            pool.free_all_blocks()
        return values.astype(cp.int32)

    def square(self, pop: Population, known=None, n_known: int = 0):
        """
        The self-play payoff matrix of pop (device int32, [i, j] = reward of
        i against j). Entries among the first n_known individuals are taken
        from `known` instead of being recomputed.
        """
        cp = cupy()
        n = len(pop)
        payoff = cp.zeros((n, n), dtype=cp.int32)
        if n_known:
            payoff[:n_known, :n_known] = known
        if n == 0 or n_known == n:
            return payoff
        rules = extract_rules(pop.heap, pop.offset, pop.length)
        zero_sum = self.reward.zero_sum
        side = max(1, isqrt(self.batch // 2))
        for i0 in range(0, n, side):
            i1 = min(i0 + side, n)
            for j0 in range(i0, n, side):
                j1 = min(j0 + side, n)
                if j1 <= n_known:  # i0 <= j0, so the whole tile is known
                    continue
                ii = cp.repeat(cp.arange(i0, i1, dtype=cp.int64), j1 - j0)
                jj = cp.tile(cp.arange(j0, j1, dtype=cp.int64), i1 - i0)
                need = ~((ii < n_known) & (jj < n_known))
                ii, jj = ii[need], jj[need]
                paired = j0 != i0 and not zero_sum
                a = cp.concatenate([ii, jj]) if paired else ii
                b = cp.concatenate([jj, ii]) if paired else jj
                values = self._score(pop, rules, a, b)
                payoff[a, b] = values
                if j0 != i0 and zero_sum:
                    payoff[jj, ii] = -values
        return payoff

    def extend(self, payoff, pop: Population, n_old: int):
        """The square matrix of pop, whose first n_old individuals are the old matrix's."""
        return self.square(pop, payoff, n_old)

    def block(self, ref: Population, pop: Population):
        """The (len(ref), len(pop)) payoff block of ref against pop (device int32)."""
        cp = cupy()
        both = Population.concat([ref, pop])
        out = cp.zeros((len(ref), len(pop)), dtype=cp.int32)
        if not len(ref) or not len(pop):
            return out
        rules = extract_rules(both.heap, both.offset, both.length)
        rows = max(1, min(len(ref), isqrt(self.batch)))
        cols = max(1, min(len(pop), self.batch // rows))
        for i0 in range(0, len(ref), rows):
            i1 = min(i0 + rows, len(ref))
            for j0 in range(0, len(pop), cols):
                j1 = min(j0 + cols, len(pop))
                a = cp.repeat(cp.arange(i0, i1, dtype=cp.int64), j1 - j0)
                b = cp.tile(cp.arange(j0, j1, dtype=cp.int64), i1 - i0) + len(ref)
                out[i0:i1, j0:j1] = self._score(both, rules, a, b).reshape(i1 - i0, j1 - j0)
        return out

    def matrix(self, ref: Sequence[Sequence[int]], pop: Sequence[Sequence[int]]) -> np.ndarray:
        """block() for host genomes, returned on the host: for tools that hold genomes as lists."""
        cp = cupy()
        return cp.asnumpy(self.block(_upload(ref), _upload(pop))).astype(np.int64)

    def close(self) -> None:
        pass

    def __enter__(self) -> "PayoffEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _upload(genomes) -> Population:
    genomes = [list(g) for g in genomes]
    n = len(genomes)
    return Population.from_genomes(genomes, np.arange(n), "random", [[] for _ in range(n)], 0)

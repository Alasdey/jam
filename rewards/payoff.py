from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
from math import isqrt
import random
from typing import Callable, List

import numpy as np

from config import ExperimentConfig
from core.matchups import HostMatchups, block
from core.types import Program
from interpreters import build_cuda_interpreter, build_interpreter
from rewards.base import RewardSpec

# These will be "per-process" globals in worker processes
_INTERP = None
_REWARD_FN: Callable | None = None


def _init_worker(cfg: ExperimentConfig, reward_fn: Callable):
    """
    Called once in each worker process.
    Builds the interpreter and stores the reward function as per-process globals.
    """
    global _INTERP, _REWARD_FN
    _INTERP = build_interpreter(cfg)  # interpreter built ONCE per worker
    _REWARD_FN = reward_fn            # top-level function, picklable


def _score(reward_fn: Callable, matchups: Callable, ref: List[Program], pop: List[Program]) -> np.ndarray:
    """The ref x pop payoff block: reward_fn on one batch of all its matchups."""
    if not ref or not pop:
        return np.zeros((len(ref), len(pop)), dtype=int)
    values = np.asarray(reward_fn(matchups(*block(ref, pop))))
    if values.shape != (len(ref) * len(pop),):
        raise ValueError(
            f"reward returned shape {values.shape}; it must return one value per "
            f"matchup ({len(ref) * len(pop)},)"
        )
    return values.astype(int).reshape(len(ref), len(pop))


def _tiles(ref: List[Program], pop: List[Program], batch: int):
    """
    Batches of about `batch` matchups covering the ref x pop block, as
    (i, j, n_rows, n_cols, paired, programs, a, b). In a square self-play
    block (ref is pop), tile (I, J) comes with its transpose (J, I) in one
    batch (paired): they need the same executions, so every ordered pair
    still runs once.
    """
    square = ref is pop
    ids: dict = {}  # distinct programs once for the whole block

    def index(population):
        return np.fromiter((ids.setdefault(tuple(code), len(ids)) for code in population),
                           dtype=np.int64, count=len(population))

    ref_ids = index(ref)
    pop_ids = ref_ids if square else index(pop)
    programs = list(ids)
    side = max(1, isqrt(batch // 2 if square else batch))
    for i in range(0, len(ref), side):
        for j in range(i if square else 0, len(pop), side):
            used = np.unique(np.concatenate([ref_ids[i:i + side], pop_ids[j:j + side]]))
            r = np.searchsorted(used, ref_ids[i:i + side])  # tile-local program ids
            c = np.searchsorted(used, pop_ids[j:j + side])
            a, b = np.repeat(r, len(c)), np.tile(c, len(r))
            paired = square and j != i
            if paired:
                a = np.concatenate([a, np.repeat(c, len(r))])
                b = np.concatenate([b, np.tile(r, len(c))])
            yield i, j, len(r), len(c), paired, [programs[k] for k in used.tolist()], a, b


def _score_batch(reward_fn: Callable, matchups: Callable, programs, a, b) -> np.ndarray:
    values = np.asarray(reward_fn(matchups(programs, a, b)))
    if values.shape != (len(a),):
        raise ValueError(f"reward returned shape {values.shape}; it must return one value per matchup ({len(a)},)")
    return values.astype(int)


def _place(payoff: np.ndarray, i, j, n_rows, n_cols, paired, values) -> None:
    n = n_rows * n_cols
    payoff[i:i + n_rows, j:j + n_cols] = values[:n].reshape(n_rows, n_cols)
    if paired:
        payoff[j:j + n_cols, i:i + n_rows] = values[n:].reshape(n_cols, n_rows)


def _init_gpu_worker(cfg: ExperimentConfig, reward_fn: Callable):
    """Spawned GPU worker: its own CUDA context and batched interpreter."""
    global _INTERP, _REWARD_FN
    _INTERP = build_cuda_interpreter(cfg)
    _REWARD_FN = reward_fn


def _gpu_batch(args):
    i, j, n_rows, n_cols, paired, programs, a, b = args
    return i, j, n_rows, n_cols, paired, _score_batch(_REWARD_FN, _INTERP.matchups, programs, a, b)


def _compute_tile(args):
    """
    Worker function that scores one tile of the block, rows ref x cols pop.
    Uses per-process globals _INTERP and _REWARD_FN.
    """
    row, col, ref, pop = args
    return row, col, _score(_REWARD_FN, partial(HostMatchups, _INTERP), ref, pop)


def _tile_shape(n_rows: int, n_cols: int, size: int) -> tuple[int, int]:
    """Near-square tiles of about `size` matchups: a task ships rows + cols programs."""
    rows = min(n_rows, max(1, isqrt(size)))
    return rows, min(n_cols, max(1, size // rows))


class PayoffEngine:
    """
    Single choke point for matchup evaluation, shared by the evolution loop
    and the offline tournament tool.

    Rewards score batches of matchups (core/matchups.py), so every backend
    runs the same reward function: "cpu" on the configured interpreter, the
    block cut into tiles of about chunksize matchups over n_workers processes
    (the ProcessPoolExecutor is created once and reused across calls); "cuda"
    on the GPU, one batch per block, with n_workers and chunksize unused.
    """

    def __init__(self, exp_cfg: ExperimentConfig, reward: RewardSpec):
        self.exp_cfg = exp_cfg
        self.reward = reward
        self._backend = exp_cfg.payoff.backend
        self._n_workers = exp_cfg.payoff.n_workers
        self._chunksize = exp_cfg.payoff.chunksize
        if self._backend not in ("cpu", "cuda"):
            raise ValueError(f"payoff.backend must be 'cpu' or 'cuda', not {self._backend!r}")
        if self._n_workers < 1:
            raise ValueError("payoff.n_workers must be at least 1")
        if self._chunksize < 1:
            raise ValueError("payoff.chunksize must be at least 1")
        self._interp = None
        self._executor = None
        if self._backend == "cuda" and exp_cfg.payoff.gpu_workers > 1:
            # Several host processes share the GPU: the host side of a batch
            # (deduplication, bookkeeping) is most of its time. Spawned, as
            # forking a process that holds a CUDA context is unsafe.
            self._executor = ProcessPoolExecutor(
                max_workers=exp_cfg.payoff.gpu_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_gpu_worker,
                initargs=(exp_cfg, reward.fn),
            )
        elif self._backend == "cuda":
            self._interp = build_cuda_interpreter(exp_cfg)
        elif self._n_workers > 1:
            self._executor = ProcessPoolExecutor(
                max_workers=self._n_workers,
                initializer=_init_worker,
                initargs=(exp_cfg, reward.fn),
            )
        else:
            self._interp = build_interpreter(exp_cfg)

    def matrix(self, ref: List[Program], pop: List[Program]) -> np.ndarray:
        """
        Payoff matrix of shape (len(ref), len(pop)); entry [i, j] is the reward
        for ref[i] playing against pop[j].
        """
        if self._backend == "cuda":
            payoff = np.zeros((len(ref), len(pop)), dtype=int)
            if not ref or not pop:
                return payoff
            tiles = _tiles(ref, pop, self.exp_cfg.payoff.gpu_batch_matchups)
            if self._executor is not None:
                for i, j, n_rows, n_cols, paired, values in self._executor.map(_gpu_batch, tiles):
                    _place(payoff, i, j, n_rows, n_cols, paired, values)
            else:
                for i, j, n_rows, n_cols, paired, programs, a, b in tiles:
                    values = _score_batch(self.reward.fn, self._interp.matchups, programs, a, b)
                    _place(payoff, i, j, n_rows, n_cols, paired, values)
            return payoff
        if self._executor is None:
            return _score(self.reward.fn, partial(HostMatchups, self._interp), ref, pop)

        payoff = np.zeros((len(ref), len(pop)), dtype=int)
        rows, cols = _tile_shape(len(ref), len(pop), self._chunksize)
        tiles = [
            (i, j, ref[i:i + rows], pop[j:j + cols])
            for i in range(0, len(ref), rows)
            for j in range(0, len(pop), cols)
        ]
        # Reproducible scheduling without consuming the evolution RNG.
        random.Random(0).shuffle(tiles)
        for i, j, values in self._executor.map(_compute_tile, tiles):
            payoff[i:i + values.shape[0], j:j + values.shape[1]] = values
        return payoff

    def extend(self, payoff: np.ndarray, old: List[Program], new: List[Program]) -> np.ndarray:
        """
        Extend a square self-play payoff matrix over `old` to cover old + new.
        The reverse block is derived as -old_new.T when the reward is zero-sum,
        computed explicitly otherwise.
        """
        new_new = self.matrix(new, new)
        if not old:
            return new_new
        old_new = self.matrix(old, new)
        new_old = -old_new.T if self.reward.zero_sum else self.matrix(new, old)
        return np.block([
            [payoff, old_new],
            [new_old, new_new],
        ])

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __enter__(self) -> "PayoffEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

import random

import numpy as np
import pytest

from config import ExperimentConfig, PayoffConfig, TreemoConfig
from creation.treemo import gen_tree
from interpreters import build_interpreter
from rewards.base import REWARDS, RewardSpec, build_reward
from rewards.payoff import PayoffEngine


def _cfg(n_workers: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        interpreter="treemo_py",
        reward="placeholder",
        treemo=TreemoConfig(max_step=10, tree_size=20),
        payoff=PayoffConfig(n_workers=n_workers),
    )


def _programs(n, seed):
    random.seed(seed)
    return [gen_tree(random.randint(5, 20)) for _ in range(n)]


def reward_row_length(interpreter, code_a, code_b) -> int:
    # deliberately NOT zero-sum: depends on code_a only
    return len(code_a) % 3


def test_matrix_matches_direct_evaluation():
    cfg = _cfg()
    ref = _programs(4, seed=31)
    pop = _programs(5, seed=37)
    with PayoffEngine(cfg, build_reward(cfg)) as engine:
        got = engine.matrix(ref, pop)
    interp = build_interpreter(cfg)
    fn = REWARDS["placeholder"].fn
    expected = np.array([[fn(interp, a, b) for b in pop] for a in ref])
    assert got.shape == (4, 5)
    assert np.array_equal(got, expected)


def test_extend_builds_square_self_play_matrix():
    cfg = _cfg()
    first = _programs(3, seed=41)
    second = _programs(4, seed=43)
    with PayoffEngine(cfg, build_reward(cfg)) as engine:
        payoff = engine.extend(np.zeros((0, 0), dtype=int), [], first)
        assert np.array_equal(payoff, engine.matrix(first, first))

        extended = engine.extend(payoff, first, second)
        assert extended.shape == (7, 7)
        assert np.array_equal(extended[:3, :3], payoff)
        old_new = engine.matrix(first, second)
        assert np.array_equal(extended[:3, 3:], old_new)
        # zero-sum: reverse block derived, not recomputed
        assert np.array_equal(extended[3:, :3], -old_new.T)
        assert np.array_equal(extended[3:, 3:], engine.matrix(second, second))


def test_extend_computes_reverse_block_when_not_zero_sum():
    cfg = _cfg()
    spec = RewardSpec("row_length", reward_row_length, zero_sum=False)
    first = _programs(3, seed=47)
    second = _programs(2, seed=53)
    with PayoffEngine(cfg, spec) as engine:
        payoff = engine.matrix(first, first)
        extended = engine.extend(payoff, first, second)
        assert np.array_equal(extended[3:, :3], engine.matrix(second, first))
        assert not np.array_equal(extended[3:, :3], -extended[:3, 3:].T)


def test_parallel_matches_sequential():
    ref = _programs(4, seed=59)
    pop = _programs(4, seed=61)
    seq_cfg = _cfg(n_workers=1)
    par_cfg = _cfg(n_workers=2)
    with PayoffEngine(seq_cfg, build_reward(seq_cfg)) as seq:
        expected = seq.matrix(ref, pop)
    with PayoffEngine(par_cfg, build_reward(par_cfg)) as par:
        got = par.matrix(ref, pop)
        # executor is reused across calls
        got2 = par.matrix(ref, pop)
    assert np.array_equal(got, expected)
    assert np.array_equal(got2, expected)

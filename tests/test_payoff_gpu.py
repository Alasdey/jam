import random

import numpy as np
import pytest

from config import ExperimentConfig, PayoffConfig, TreemoConfig
from core.matchups import block
from creation.treemo import gen_tree
from rewards.base import REWARDS, RewardSpec, build_reward
from rewards.payoff import PayoffEngine


def kitchen_sink(m):
    """Every Matchups operation, so both backends must agree on all of them."""
    out = m.run(m.a, m.b)
    staple = m.const([0, 1, 2, 3, 4, 5])  # not 0/1: stays on the host
    back = m.run(out, m.a)  # an output used as a program
    again = m.run(m.b, out)  # an output used as an input
    both = m.a.concat(m.b)
    score = out.length() - again.length() + out.count(1) - back.count(0) + staple.count(3)
    score += out.lcs(both) % 7 + m.run(m.a, staple).length() + (10 * both.similarity(m.b)).astype(int)
    score += np.where(out.equals(m.run(m.b, m.a)), 100, 0) + m.memory(m.a, m.b).length()
    score += m.run(both, m.a).length() % 5 + m.run(m.a, staple).concat(out).count(2)
    return score + m.map(lambda x, y, k: (len(x) * 3 + sum(y) + k) % 11, out, m.b, out.length())


SPECS = [REWARDS[name] for name in sorted(REWARDS)] + [RewardSpec("kitchen_sink", kitchen_sink, zero_sum=False)]


def _cfg(reward: str, backend: str, interpreter: str = "treemo", batch: int = 1_000_000) -> ExperimentConfig:
    return ExperimentConfig(
        interpreter=interpreter,
        reward=reward,
        treemo=TreemoConfig(max_step=50, tree_size=30),
        payoff=PayoffConfig(n_workers=1, backend=backend, gpu_batch_matchups=batch),
    )


def _programs(n, seed):
    random.seed(seed)
    progs = [gen_tree(random.randint(2, 60)) for _ in range(n)]
    return progs + progs[:3] + [[], [1, 0]]  # duplicates must score like their originals


@pytest.mark.cuda
@pytest.mark.parametrize("batch", [1_000_000, 37], ids=["one_batch", "tiled"])
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_cuda_matches_cpu_for_every_reward(spec, batch):
    ref = _programs(20, seed=1)
    pop = _programs(25, seed=2) + ref[:4]
    with PayoffEngine(_cfg("blind", "cpu"), spec) as cpu, \
         PayoffEngine(_cfg("blind", "cuda", batch=batch), spec) as gpu:
        for a, b in ((ref, pop), (pop, pop), (pop, ref), ([], pop), (ref, [])):
            expected = cpu.matrix(a, b)
            got = gpu.matrix(a, b)
            assert got.shape == expected.shape
            assert np.array_equal(got, expected)
        old = np.zeros((0, 0), dtype=int)
        assert np.array_equal(gpu.extend(gpu.extend(old, [], ref), ref, pop),
                              cpu.extend(cpu.extend(old, [], ref), ref, pop))


@pytest.mark.cuda
def test_square_blocks_run_each_ordered_pair_once():
    progs = _programs(30, seed=3)
    cfg = _cfg("quine_pressure", "cuda")
    with PayoffEngine(cfg, build_reward(cfg)) as gpu:
        m = gpu._interp.matchups(*block(progs, progs))
        REWARDS["quine_pressure"].fn(m)
    distinct = len({tuple(p) for p in progs})
    assert m.stats["gpu_runs"] + m.stats["cpu_runs"] == distinct ** 2


def test_cuda_backend_needs_treemo():
    cfg = _cfg("quine_pressure", "cuda", interpreter="treemo_py")
    with pytest.raises(ValueError, match="treemo"):
        PayoffEngine(cfg, build_reward(cfg))


def test_unknown_backend_rejected():
    cfg = _cfg("quine_pressure", "tpu")
    with pytest.raises(ValueError, match="backend"):
        PayoffEngine(cfg, build_reward(cfg))

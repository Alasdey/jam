import random

import numpy as np
import pytest

from config import ExperimentConfig, PayoffConfig, TreemoConfig
from core.matchups import HostMatchups, block, play_one, zipped
from creation.treemo import gen_tree
from interpreters.iconfractran.iconfractran import IconfractranInterpreter
from interpreters.treemo_c.treemo import TreemoInterpreter as CTreemo
from rewards.base import REWARDS, RewardSpec
from rewards.payoff import PayoffEngine


class CountingInterpreter:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def run(self, code, inp):
        self.calls += 1
        return self.inner.run(code, inp)


def _programs(n, seed):
    random.seed(seed)
    progs = [gen_tree(random.randint(2, 40)) for _ in range(n)]
    return progs + progs[:2]


def asymmetric(m):
    """Not zero-sum, and touches outputs, constants, concatenation and map."""
    out = m.run(m.a, m.b)
    tail = m.run(m.a, m.const([1, 0])).concat(out)
    return out.lcs(m.b) - tail.count(1) + m.map(lambda x, n: (sum(x) + n) % 5, out, m.a.length())


def test_each_distinct_execution_runs_once():
    progs = _programs(12, seed=1)
    it = CountingInterpreter(CTreemo(50, False, False))
    m = HostMatchups(it, *block(progs, progs))
    first = m.run(m.a, m.b)
    m.run(m.b, m.a)  # every ordered pair already ran, transposed
    assert it.calls == len({tuple(p) for p in progs}) ** 2
    direct = m.map(lambda out, a, b: out == it.inner.run(a, b)[0], first, m.a, m.b)
    assert direct.all()


def test_map_hands_each_matchup_fresh_lists_and_entries():
    progs = _programs(4, seed=2)
    m = HostMatchups(CTreemo(), *block(progs, progs))
    grown = m.map(lambda code, n: (code.append(1), len(code) - n)[1], m.a, m.a.length())
    assert grown.tolist() == [1] * m.n  # an append in one call is not seen by the next
    with pytest.raises(ValueError, match="one entry per matchup"):
        m.map(len, np.zeros(m.n + 1))


def test_tapes_belong_to_their_batch():
    progs = _programs(3, seed=3)
    m1 = HostMatchups(CTreemo(), *block(progs, progs))
    m2 = HostMatchups(CTreemo(), *block(progs, progs))
    with pytest.raises(ValueError, match="different batch"):
        m1.run(m1.a, m2.b)


def test_memory_is_the_interpreters_second_result():
    class Doubler:
        def run(self, code, inp):
            return list(inp), list(inp) * 2

    m = HostMatchups(Doubler(), *zipped([[3], [4, 5]], [[7, 8], [9]]))
    assert m.memory(m.a, m.b).length().tolist() == [4, 2]
    assert m.memory(m.a, m.b).equals(m.b.concat(m.b)).all()


@pytest.mark.parametrize("chunksize", [1, 7, 50])
def test_parallel_tiles_match_one_batch(chunksize):
    ref, pop = _programs(9, seed=4), _programs(11, seed=5)
    spec = RewardSpec("asymmetric", asymmetric, zero_sum=False)

    def engine(n_workers):
        cfg = ExperimentConfig(treemo=TreemoConfig(max_step=20, tree_size=15),
                               payoff=PayoffConfig(n_workers=n_workers, chunksize=chunksize))
        return PayoffEngine(cfg, spec)

    with engine(1) as one, engine(2) as tiles:
        assert np.array_equal(tiles.matrix(ref, pop), one.matrix(ref, pop))


def test_rewards_returning_the_wrong_shape_are_refused():
    cfg = ExperimentConfig(payoff=PayoffConfig(n_workers=1))
    with PayoffEngine(cfg, RewardSpec("scalar", lambda m: 1, zero_sum=True)) as engine:
        with pytest.raises(ValueError, match="one value per matchup"):
            engine.matrix(_programs(2, seed=6), _programs(2, seed=7))


def test_rewards_run_on_any_interpreter():
    rng = random.Random(8)
    progs = [[rng.randint(-20, 40) for _ in range(rng.randint(0, 12))] for _ in range(6)]
    it = IconfractranInterpreter(max_step=30)

    def similarity(out, ref):
        common = [[0] * (len(ref) + 1) for _ in range(len(out) + 1)]
        for i, x in enumerate(out):
            for j, y in enumerate(ref):
                common[i + 1][j + 1] = common[i][j] + 1 if x == y else max(common[i][j + 1], common[i + 1][j])
        return common[-1][-1] / len(ref) if out and ref else 0.0

    for a in progs:
        for b in progs:
            expected = np.sign(similarity(it.run(a, b)[0], a) - similarity(it.run(b, a)[0], b))
            assert play_one(REWARDS["quine_pressure"].fn, it, a, b) == expected

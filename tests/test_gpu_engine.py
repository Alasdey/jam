import random

import numpy as np
import pytest

import interpreters.treemo_gpu.rules as rules_module
from config import ExperimentConfig
from core.device import cupy
from core.matchups import Matchups, lcs_length
from core.population import Population
from tests.reference.creation_cpu import gen_tree
from interpreters.treemo_c.treemo import TreemoInterpreter as CTreemo
from interpreters.treemo_gpu.rules import extract_rules
from interpreters.treemo_gpu.treemo import TreemoRuntime
from interpreters.treemo_py.treemo import tree_to_rules
from rewards.base import REWARDS, RewardSpec
from rewards.payoff import PayoffEngine
from tests.reference.host_matchups import HostMatchups, block

pytestmark = pytest.mark.cuda


def _programs(n, seed, max_nodes=60):
    rng = random.Random(seed)
    random.seed(seed)  # gen_tree draws from the global RNG
    progs = [gen_tree(rng.randint(1, max_nodes)) for _ in range(n)]
    # forests: the implicit root then has several children
    progs += [sum((gen_tree(rng.randint(1, 10)) for _ in range(rng.randint(2, 4))), [])
              for _ in range(n // 4)]
    return progs + [[], [1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0, 1, 0]]


def _pop(genomes):
    n = len(genomes)
    return Population.from_genomes(genomes, np.arange(n), "random", [[] for _ in range(n)], 0)


def _engine(reward="blind", batch=1_000_000, memory_mb=1024, **treemo):
    cfg = ExperimentConfig()
    cfg.reward = reward
    cfg.payoff.gpu_batch_matchups = batch
    cfg.payoff.gpu_memory_mb = memory_mb
    for key, value in treemo.items():
        setattr(cfg.treemo, key, value)
    return PayoffEngine(cfg, REWARDS[reward])


def _reference_block(reward_fn, ref, pop, **treemo):
    t = ExperimentConfig().treemo
    settings = dict(max_step=t.max_step, pass_mode=t.pass_mode, first_mode=t.first_mode) | treemo
    interp = CTreemo(**settings)
    values = np.asarray(reward_fn(HostMatchups(interp, *block(ref, pop))))
    return values.astype(int).reshape(len(ref), len(pop))


@pytest.mark.parametrize("chunk", [1 << 24, 997])
def test_rules_match_the_reference(chunk, monkeypatch):
    monkeypatch.setattr(rules_module, "_CHUNK_BITS", chunk)
    progs = _programs(300, seed=1, max_nodes=120)
    pop = _pop(progs)
    t = extract_rules(pop.heap, pop.offset, pop.length)
    table, ident, start, count = t.table.get(), t.ident.get(), t.start.get(), t.count.get()
    for p, code in enumerate(progs):
        rows = table[start[p]:start[p] + count[p]].tolist()
        want = [(list(x), list(y)) for x, y in tree_to_rules(code)]
        assert [(code[a:a + b], code[c:c + d]) for a, b, c, d in rows] == want
        assert ident[start[p]:start[p] + count[p]].tolist() == [int(x == y) for x, y in want]


def test_unbalanced_programs_are_refused():
    pop = _pop([[1, 1, 0]])
    with pytest.raises(ValueError, match="balanced"):
        extract_rules(pop.heap, pop.offset, pop.length)


@pytest.mark.parametrize("max_step,pass_mode,first_mode",
                         [(50, False, False), (50, True, False), (50, True, True), (50, False, True),
                          (7, True, False), (0, False, False)])
def test_runs_match_c(max_step, pass_mode, first_mode):
    progs = _programs(60, seed=2)
    rng = random.Random(3)
    pairs = [(rng.randrange(len(progs)), rng.randrange(len(progs))) for _ in range(800)]
    pop = _pop(progs)
    runtime = TreemoRuntime(max_step, pass_mode, first_mode)
    m = Matchups(runtime, pop, [a for a, _ in pairs], [b for _, b in pairs])
    out = m.run(m.a, m.b)
    got = m._store.download(out.ids)
    c = CTreemo(max_step, pass_mode, first_mode)
    assert got == [c.run(progs[a], progs[b])[0] for a, b in pairs]
    assert out.length().get().tolist() == [len(g) for g in got]


def test_states_that_grow_far_past_their_input():
    grower = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0]  # a rule replacing one leaf by three
    pop = _pop([grower, [1, 0] * 3])
    m = Matchups(TreemoRuntime(50, False, False), pop, [0], [1])
    got = m._store.download(m.run(m.a, m.b).ids)
    assert got == [CTreemo(50, False, False).run(grower, [1, 0] * 3)[0]]


def test_small_memory_budget_splits_launches():
    progs = _programs(40, seed=4)
    pop = _pop(progs)
    a = np.repeat(np.arange(len(progs)), len(progs))
    b = np.tile(np.arange(len(progs)), len(progs))
    big = Matchups(TreemoRuntime(50, False, False, memory_mb=1024), pop, a, b)
    small = Matchups(TreemoRuntime(50, False, False, memory_mb=1), pop, a, b)
    assert big._store.download(big.run(big.a, big.b).ids) == small._store.download(small.run(small.a, small.b).ids)


@pytest.mark.parametrize("name", sorted(REWARDS))
def test_every_reward_matches_the_host_reference(name):
    progs = _programs(24, seed=5)
    want = _reference_block(REWARDS[name].fn, progs, progs)
    got = _engine(name).square(_pop(progs)).get()
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("name", sorted(REWARDS))
def test_tiles_and_extension_change_nothing(name):
    progs = _programs(30, seed=6)
    pop = _pop(progs)
    full = _engine(name).square(pop).get()
    assert _engine(name, batch=37).square(pop).get().tolist() == full.tolist()
    n_old = 17
    known = cupy().asarray(full[:n_old, :n_old])
    assert _engine(name, batch=50).extend(known, pop, n_old).get().tolist() == full.tolist()


def test_block_matches_the_reference():
    ref, pop = _programs(12, seed=7), _programs(9, seed=8)
    want = _reference_block(REWARDS["blind"].fn, ref, pop)
    engine = _engine("blind", batch=40)
    assert engine.matrix(ref, pop).tolist() == want.tolist()


def test_each_distinct_execution_runs_once():
    progs = _programs(10, seed=9)
    pop = _pop(progs)
    n = len(progs)
    m = Matchups(TreemoRuntime(50, False, False), pop, np.repeat(np.arange(n), n), np.tile(np.arange(n), n))
    m.run(m.a, m.b)
    m.run(m.b, m.a)  # the same ordered pairs again
    assert m.stats["runs"] == n * n


def test_lcs_on_device_and_long_tapes_on_the_host(capsys):
    rng = random.Random(10)
    lengths = [0, 1, 31, 32, 33, 100, 2048, 4096, 5000, 16384, 17000]
    tapes = [[rng.randint(0, 1) for _ in range(n)] for n in lengths]
    pop = _pop(tapes)
    k = len(tapes)
    a, b = np.repeat(np.arange(k), k), np.tile(np.arange(k), k)
    m = Matchups(TreemoRuntime(50, False, False), pop, a, b)
    got = m.a.lcs(m.b).get().tolist()
    assert got == [lcs_length(tapes[i], tapes[j]) for i, j in zip(a, b)]
    assert "SLOW HOST PATH" in capsys.readouterr().err  # 17000 vs 17000


def test_map_warns_loudly_and_matches(capsys):
    progs = _programs(6, seed=11)
    pop = _pop(progs)
    n = len(progs)
    m = Matchups(TreemoRuntime(50, False, False), pop, np.repeat(np.arange(n), n), np.tile(np.arange(n), n))
    got = m.map(lambda x, y: len(x) * 1000 + len(y), m.a, m.b).get().tolist()
    assert got == [len(progs[i]) * 1000 + len(progs[j]) for i in range(n) for j in range(n)]
    assert "SLOW HOST PATH" in capsys.readouterr().err


def test_const_must_be_binary_and_tapes_belong_to_their_batch():
    pop = _pop(_programs(3, seed=12))
    m1 = Matchups(TreemoRuntime(50, False, False), pop, [0, 1], [1, 2])
    m2 = Matchups(TreemoRuntime(50, False, False), pop, [0, 1], [1, 2])
    with pytest.raises(ValueError, match="0 and 1"):
        m1.const([0, 2])
    with pytest.raises(ValueError, match="different batch"):
        m1.run(m1.a, m2.b)


def test_rewards_returning_the_wrong_shape_are_refused():
    spec = RewardSpec("bad", lambda m: np.zeros(3), zero_sum=True)
    engine = PayoffEngine(ExperimentConfig(), spec)
    with pytest.raises(ValueError, match="one value per matchup"):
        engine.square(_pop(_programs(4, seed=13)))


def _every_operation(m):
    out = m.run(m.a, m.b)
    both = out.concat(m.memory(m.b, m.a))
    staple = m.run(m.a, m.const([1, 0, 1, 1, 0, 0]))
    return (both.count(1) - 2 * both.count(0) + 3 * out.lcs(m.b) + (out.similarity(m.a) * 100).astype(int)
            + 5 * staple.length() + 7 * out.equals(m.b) + both.count(5))


def test_every_operation_matches_the_host_reference():
    progs = _programs(16, seed=14)
    spec = RewardSpec("every_operation", _every_operation, zero_sum=False)
    want = _reference_block(_every_operation, progs, progs)
    cfg = ExperimentConfig()
    cfg.payoff.gpu_batch_matchups = 60
    assert PayoffEngine(cfg, spec).square(_pop(progs)).get().tolist() == want.tolist()

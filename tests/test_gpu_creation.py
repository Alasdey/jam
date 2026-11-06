import random
from collections import Counter

import numpy as np
import pytest

import creation.treemo as creation_module
from config import ExperimentConfig
from core.device import launch
from core.population import METHODS, Population
from core.rng import DeviceRng
from creation.treemo import TreemoCreator
from interpreters.treemo_c.treemo import TreemoInterpreter as CTreemo
from interpreters.treemo_gpu.treemo import TreemoRuntime
from tests.reference import creation_cpu as ref

pytestmark = pytest.mark.cuda


def _cfg(rate=0.05, crossover=0.8, homoiconic=0.3, mutation="subtree", cross="random_depth", tree_size=100):
    cfg = ExperimentConfig()
    cfg.genetics.mutation_rate = rate
    cfg.genetics.crossover_prob = crossover
    cfg.genetics.homoiconic_prob = homoiconic
    cfg.genetics.tree_mutation_op = mutation
    cfg.genetics.tree_crossover_op = cross
    cfg.treemo.tree_size = tree_size
    return cfg


def _pop(genomes):
    n = len(genomes)
    return Population.from_genomes(genomes, np.arange(100, 100 + n), "random", [[] for _ in range(n)], 0)


def _gpu_children(cfg, parents, n_offspring, n_random=0, seed=0):
    creator = TreemoCreator(cfg, TreemoRuntime.from_config(cfg))
    pop = creator.create(_pop(parents), n_random, n_offspring, DeviceRng(seed, 1), gen=1, first_id=1000)
    return pop, [(METHODS[m], tuple(g)) for m, g in zip(pop.method, pop.genomes())]


class _CpuCreator:
    """creation/treemo.py TreemoCreator before the port, on the reference operators."""

    def __init__(self, cfg):
        g = cfg.genetics
        self.rate = g.mutation_rate
        self.mutate_fn = {"subtree": ref.mutate_tree_subtree, "leaf": ref.mutate_tree_leaf}[g.tree_mutation_op]
        self.cross_fn = {"random_depth": ref.crossover_tree_random_depth,
                         "depth1": ref.crossover_tree_depth1}[g.tree_crossover_op]

    def mutate(self, p):
        return self.mutate_fn(p, self.rate)

    def crossover(self, a, b):
        return self.cross_fn(a, b)

    def homoiconic(self, interp, a, b):
        out = interp.run(a, b)[0]
        return out if out else None


def _cpu_children(cfg, parents, n_offspring, seed=0):
    random.seed(seed)
    t = cfg.treemo
    interp = CTreemo(t.max_step, t.pass_mode, t.first_mode)
    kids, methods, _ = ref.make_offspring(_CpuCreator(cfg), parents, n_offspring, cfg.genetics, interp)
    return [(m, tuple(k)) for m, k in zip(methods, kids)]


def _total_variation(xs, ys):
    cx, cy = Counter(xs), Counter(ys)
    return 0.5 * sum(abs(cx[k] / len(xs) - cy[k] / len(ys)) for k in cx.keys() | cy.keys())


def _assert_same_distribution(cfg, parents, n):
    """GPU children differ from CPU children no more than two CPU samples differ from each other."""
    _, gpu = _gpu_children(cfg, parents, n)
    cpu, cpu2 = _cpu_children(cfg, parents, n, seed=1), _cpu_children(cfg, parents, n, seed=2)
    floor = _total_variation(cpu, cpu2)  # grows with the number of distinct outcomes
    assert _total_variation(gpu, cpu) < 1.3 * floor + 0.01
    return gpu, cpu


SMALL = [[1, 1, 0, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]]


def test_random_trees_match_the_cpu_distribution():
    # Not uniform over the 14 plane trees with 5 nodes: the CPU gen_tree
    # weights a tree by the length of its first top-level subtree.
    _, kids = _gpu_children(_cfg(tree_size=5), [], 0, n_random=42_000)
    gpu = [g for _, g in kids]
    random.seed(0)
    cpu = [tuple(ref.gen_tree(5)) for _ in range(42_000)]
    assert len(set(gpu)) == 14 and all(ref._is_balanced(list(g)) and len(g) == 8 for g in gpu)
    assert _total_variation(gpu, cpu) < 0.02


@pytest.mark.parametrize("mutation,rate", [("subtree", 0.5), ("subtree", 1.0), ("leaf", 0.7)])
def test_mutations_match_the_cpu_distribution(mutation, rate):
    _assert_same_distribution(_cfg(rate=rate, crossover=0.0, mutation=mutation), SMALL[:1], 40_000)


@pytest.mark.parametrize("cross", ["random_depth", "depth1"])
def test_crossovers_match_the_cpu_distribution(cross):
    _assert_same_distribution(_cfg(crossover=1.0, homoiconic=0.0, cross=cross), SMALL, 40_000)


def test_full_dispatch_matches_the_cpu_distribution():
    gpu, cpu = _assert_same_distribution(_cfg(rate=0.3), SMALL, 60_000)
    gm, cm = Counter(m for m, _ in gpu), Counter(m for m, _ in cpu)
    for method in ("crossover", "homoiconic", "mutate"):
        assert abs(gm[method] - cm[method]) / 60_000 < 0.01


def test_homoiconic_children_are_run_outputs():
    random.seed(3)
    parents = [ref.gen_tree(random.randint(3, 30)) for _ in range(20)]
    cfg = _cfg(crossover=1.0, homoiconic=1.0)
    pop, kids = _gpu_children(cfg, parents, 500)
    c = CTreemo(cfg.treemo.max_step, cfg.treemo.pass_mode, cfg.treemo.first_mode)
    index = {pid: parents[i] for i, pid in enumerate(range(100, 120))}
    n_homo = 0
    for (method, child), (pa, pb) in zip(kids, pop.parents.tolist()):
        out = c.run(index[pa], index[pb])[0]
        if method == "homoiconic":
            assert list(child) == out and out
            n_homo += 1
        else:
            assert method == "crossover" and not out
    assert n_homo > 0


def test_children_are_valid_trees_with_their_parents_recorded():
    random.seed(4)
    parents = [ref.gen_tree(random.randint(50, 400)) for _ in range(30)]
    for mutation in ("subtree", "leaf"):
        for cross in ("random_depth", "depth1"):
            pop, kids = _gpu_children(_cfg(rate=0.2, mutation=mutation, cross=cross), parents, 2_000, n_random=50)
            assert all(ref._is_balanced(list(g)) for _, g in kids)
            for (method, _), ps in zip(kids, pop.parents.tolist()):
                n_parents = sum(p >= 0 for p in ps)
                assert n_parents == {"random": 0, "mutate": 1, "crossover": 2, "homoiconic": 2}[method]
            assert pop.ids.tolist() == list(range(1000, 1000 + len(pop)))


def test_overflow_retries_make_the_same_children(monkeypatch):
    random.seed(5)
    parents = [ref.gen_tree(random.randint(5, 40)) for _ in range(10)]
    cfg = _cfg(rate=1.0, crossover=0.0)
    _, roomy = _gpu_children(cfg, parents, 3_000, seed=9)
    monkeypatch.setattr(creation_module, "_MUTATION_ROOM", (1, 2))
    _, tight = _gpu_children(cfg, parents, 3_000, seed=9)
    assert tight == roomy


def test_large_tree_statistics_match_the_cpu():
    random.seed(6)
    parents = [ref.gen_tree(random.randint(200, 600)) for _ in range(20)]
    cfg = _cfg(rate=0.05)
    _, gpu = _gpu_children(cfg, parents, 4_000)
    cpu = _cpu_children(cfg, parents, 4_000)
    for stat in (len, lambda g: ref._tree_max_depth(list(g))):
        g = np.array([stat(k) for _, k in gpu], dtype=float)
        c = np.array([stat(k) for _, k in cpu], dtype=float)
        assert abs(g.mean() - c.mean()) < 0.05 * c.mean()
        assert abs(g.std() - c.std()) < 0.1 * c.std()


def test_bootstrap_without_survivors_makes_random_trees():
    pop, kids = _gpu_children(_cfg(tree_size=20), [], 30, n_random=10)
    assert len(kids) == 40 and all(m == "random" and len(g) == 38 for m, g in kids)


def _thread_kernel_children(monkeypatch, *args, **kwargs):
    """_gpu_children made by the one-thread-per-child kernel the warp kernel replaced."""
    with monkeypatch.context() as mp:
        mp.setattr(creation_module, "_SOURCE", "tests/reference/creation_thread.cu")
        mp.setattr(creation_module, "_CREATE_LAUNCH", launch)
        return _gpu_children(*args, **kwargs)


@pytest.mark.parametrize("mutation,cross,rate,room", [
    ("subtree", "random_depth", 0.05, None), ("subtree", "depth1", 0.4, None), ("subtree", "random_depth", 1.0, (1, 2)),
    ("leaf", "random_depth", 0.7, None), ("leaf", "depth1", 1.0, (1, 0)),
])
def test_warp_kernel_makes_the_thread_kernels_children(monkeypatch, mutation, cross, rate, room):
    # Same parents, ops and Philox counters: the very same children, retries included.
    random.seed(7)
    parents = [ref.gen_tree(random.randint(1, 900)) for _ in range(40)] + SMALL + [[], [1, 0]]
    cfg = _cfg(rate=rate, crossover=0.5, homoiconic=0.0, mutation=mutation, cross=cross, tree_size=60)
    if room is not None:
        monkeypatch.setattr(creation_module, "_MUTATION_ROOM", room)
    _, warp = _gpu_children(cfg, parents, 2_000, n_random=200, seed=11)
    _, thread = _thread_kernel_children(monkeypatch, cfg, parents, 2_000, n_random=200, seed=11)
    assert warp == thread

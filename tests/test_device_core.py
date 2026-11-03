import random

import numpy as np
import pytest

from core.device import cupy
from core.population import METHODS, Population, pack, unpack
from core.rng import DeviceRng
from tests.reference.creation_cpu import gen_tree
from tests.reference import philox

pytestmark = pytest.mark.cuda


def test_uniform_matches_the_reference_philox():
    rng = DeviceRng(seed=(7 << 32) | 12345, gen=3)
    rng.uniform(5)  # call 0
    got = cupy().asnumpy(rng.uniform(1000))  # call 1
    assert np.array_equal(got, philox.uniform((7 << 32) | 12345, 3, 1, 1000))


def test_draws_depend_only_on_seed_gen_and_call():
    a, b = DeviceRng(42, 5), DeviceRng(42, 5)
    assert np.array_equal(a.uniform(100).get(), b.uniform(100).get())
    assert not np.array_equal(a.uniform(100).get(), a.uniform(100).get())  # successive calls differ
    assert not np.array_equal(DeviceRng(42, 5).uniform(100).get(), DeviceRng(42, 6).uniform(100).get())
    assert not np.array_equal(DeviceRng(42, 5).uniform(100).get(), DeviceRng(43, 5).uniform(100).get())


def test_integers_cover_their_range_uniformly():
    rng = DeviceRng(1, 0)
    k = rng.integers(7, 70_000).get()
    assert k.min() == 0 and k.max() == 6
    counts = np.bincount(k, minlength=7)
    assert np.all(np.abs(counts - 10_000) < 500)
    per = rng.integers(np.array([1, 2, 3, 1000] * 100)).get().reshape(100, 4)
    assert (per[:, 0] == 0).all() and (per[:, 1] < 2).all() and (per[:, 2] < 3).all()


def test_permutation_is_a_permutation():
    p = DeviceRng(9, 2).permutation(1000).get()
    assert sorted(p.tolist()) == list(range(1000))
    assert not np.array_equal(p, np.arange(1000))


def _genomes(n, seed):
    random.seed(seed)
    return [gen_tree(random.randint(1, 80)) for _ in range(n)]


def test_pack_unpack_round_trip():
    genomes = _genomes(50, 1) + [[], [1, 0] * 40]
    words, offsets = pack(genomes)
    assert words[0] == 0
    assert unpack(words, offsets, np.array([len(g) for g in genomes])) == genomes


def test_population_take_concat_and_records():
    genomes = _genomes(30, 2)
    pop = Population.from_genomes(genomes, ids=range(100, 130), method="random",
                                  parents=[[] for _ in genomes], born_gen=4)
    assert pop.genomes() == genomes
    sub = pop.take([5, 0, 29, 5])
    assert sub.genomes() == [genomes[5], genomes[0], genomes[29], genomes[5]]
    assert sub.ids.tolist() == [105, 100, 129, 105]
    other = Population.from_genomes([[1, 0]], ids=[7], method=["mutate"], parents=[[100]], born_gen=[5])
    both = Population.concat([sub, other])
    assert both.genomes() == sub.genomes() + [[1, 0]]
    rec = both.records([4])[0]
    assert rec == {"id": 7, "genome": [1, 0], "method": "mutate", "parents": [100], "born_gen": 5}
    assert METHODS[both.method[0]] == "random"


def test_hashes_and_equality_follow_genomes():
    genomes = _genomes(40, 3)
    genomes += [list(g) for g in genomes[:10]]  # exact duplicates
    pop = Population.from_genomes(genomes, ids=range(len(genomes)), method="random",
                                  parents=[[] for _ in genomes], born_gen=0)
    h = pop.hashes().get()
    for i in range(len(genomes)):
        for j in range(len(genomes)):
            if genomes[i] == genomes[j]:
                assert h[i] == h[j]
    x = np.repeat(np.arange(len(genomes)), len(genomes))
    y = np.tile(np.arange(len(genomes)), len(genomes))
    eq = pop.equal(x, y).get()
    assert eq.tolist() == [genomes[i] == genomes[j] for i, j in zip(x, y)]


def test_empty_population():
    pop = Population.empty()
    assert len(pop) == 0 and pop.genomes() == [] and len(pop.take([])) == 0

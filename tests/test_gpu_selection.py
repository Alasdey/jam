import random

import numpy as np
import pytest

from config import (
    BestWorstRandomStepConfig, CapStepConfig, DedupeStepConfig, KNNNoveltyStepConfig, KNNScoreStepConfig,
    LexicaseStepConfig, MaxLengthStepConfig, MixtureStepConfig, NashStepConfig, SkimStepConfig,
)
from core.device import cupy
from core.population import Population
from core.rng import DeviceRng
from tests.reference.creation_cpu import gen_tree
from selection.base import SELECTION_STEPS, build_selection
from selection.best_worst_random import best_worst_random_selection
from selection.knn_novelty import _knn_novelty_distances, knn_novelty_selection
from selection.knn_score import _knn_score_distances, knn_score_selection
from selection.lexicase import _lexicase, identical_row_groups, lexicase_selection
from selection.mixture import mixture_selection
from selection.skim import iterated_elimination
from tests.reference.selection_cpu import knn_novelty as ref_novelty
from tests.reference.selection_cpu import knn_score as ref_score
from tests.reference.selection_cpu import mixture as ref_mixture
from tests.reference.selection_cpu import skim as ref_skim
from tests.reference.selection_cpu.base import build_skim as ref_build_skim

pytestmark = pytest.mark.cuda


def _payoff(n, seed, low=-1, high=2, cols=None, duplicate_rows=0):
    rng = np.random.default_rng(seed)
    p = rng.integers(low, high, size=(n, cols or n)).astype(np.int32)
    for i in range(duplicate_rows):
        p[n - 1 - i] = p[i]  # identical payoff rows
    return p


def _pop(n=0, genomes=None):
    genomes = genomes if genomes is not None else [[1, 0]] * n
    k = len(genomes)
    return Population.from_genomes(genomes, np.arange(k), "random", [[] for _ in range(k)], 0)


# ── lexicase ─────────────────────────────────────────────────────────────

def _ref_lexicase(payoff, n_accepted, order, tie):
    remaining = np.arange(len(payoff))
    selected = []
    for p in range(n_accepted):
        cand = remaining
        for c in order[p]:
            if len(cand) == 1:
                break
            s = payoff[cand, c]
            cand = cand[s == s.max()]
        k = min(int(tie[p] * len(cand)), len(cand) - 1)
        winner = np.sort(cand)[k]
        selected.append(winner)
        remaining = remaining[remaining != winner]
    return np.sort(selected)


@pytest.mark.parametrize("n,cols,low,high,dups,accepted", [
    (60, None, -1, 2, 10, 20), (200, None, -1, 2, 50, 150), (40, 7, 0, 3, 5, 30), (30, None, -50, 50, 0, 29),
    (25, 0, 0, 1, 0, 10), (50, None, 0, 1, 0, 49),  # no cases; every row identical
])
def test_lexicase_matches_the_reference_draw_for_draw(n, cols, low, high, dups, accepted):
    payoff = _payoff(n, seed=n + accepted, low=low, high=high, cols=cols, duplicate_rows=dups)
    n_cases = payoff.shape[1]
    rng = np.random.default_rng(1)
    order = np.argsort(rng.random((accepted, n_cases)), axis=1).astype(np.int32)
    tie = rng.random(accepted)
    cp = cupy()
    got = _lexicase(cp.asarray(payoff), accepted, cp.asarray(order), cp.asarray(tie)).get()
    assert got.tolist() == _ref_lexicase(payoff, accepted, order, tie).tolist()


def test_lexicase_quotas():
    payoff = cupy().asarray(_payoff(50, seed=3))
    sel = lexicase_selection(payoff, 20, DeviceRng(1, 0)).get()
    assert len(set(sel.tolist())) == 20 and (np.diff(sel) > 0).all()
    assert lexicase_selection(payoff, 60, DeviceRng(1, 0)).get().tolist() == list(range(50))
    assert len(lexicase_selection(payoff, 0, DeviceRng(1, 0))) == 0


def test_identical_rows_share_a_group():
    payoff = _payoff(30, seed=4, duplicate_rows=8)
    group_of, rep = identical_row_groups(cupy().asarray(payoff))
    group_of = group_of.get()
    for i in range(30):
        for j in range(30):
            assert (group_of[i] == group_of[j]) == bool((payoff[i] == payoff[j]).all())


# ── skim ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(5))
def test_iterated_elimination_matches_the_reference(seed):
    payoff = _payoff(80, seed=seed, low=-3, high=4)
    got = iterated_elimination(cupy().asarray(payoff)).get()
    assert got.tolist() == ref_skim.iterated_elimination_strictly_dominated_rows_fast(payoff).tolist()


@pytest.mark.parametrize("n_rounds,n_accepted", [(1, None), (3, None), (3, 70)])
def test_skim_step_matches_the_reference(n_rounds, n_accepted):
    payoff = _payoff(90, seed=7, low=-2, high=3)
    cfg = SkimStepConfig(n_rounds=n_rounds, fraction=1.0, n_accepted=n_accepted)
    got = SELECTION_STEPS["skim"](cfg)(cupy().asarray(payoff), _pop(90), DeviceRng(0, 0)).get()
    assert got.tolist() == ref_build_skim(cfg)(payoff, [[1, 0]] * 90).tolist()


def test_skim_fraction_keeps_some_dominated_rows():
    payoff = _payoff(90, seed=8, low=-2, high=3)
    payoff[45:] = payoff[:45] - 1  # rows 45.. are each dominated by an earlier row
    full = SELECTION_STEPS["skim"](SkimStepConfig(n_rounds=1, fraction=1.0, n_accepted=None))
    half = SELECTION_STEPS["skim"](SkimStepConfig(n_rounds=1, fraction=0.5, n_accepted=None))
    a = full(cupy().asarray(payoff), _pop(90), DeviceRng(0, 0)).get()
    b = half(cupy().asarray(payoff), _pop(90), DeviceRng(0, 0)).get()
    assert len(a) < 90 and set(a) < set(b) and len(b) - len(a) == (90 - len(a)) // 2


# ── kNN ──────────────────────────────────────────────────────────────────

def test_knn_distances_match_the_reference():
    payoff = _payoff(300, seed=9, low=-5, high=6)
    cp = cupy()
    for k in (1, 10, 299, 400):
        np.testing.assert_allclose(_knn_novelty_distances(cp.asarray(payoff), k).get(),
                                   ref_novelty._knn_novelty_distances(payoff.astype(float), k), rtol=1e-12)
        np.testing.assert_allclose(_knn_score_distances(cp.asarray(payoff), k).get(),
                                   ref_score._knn_score_distances(payoff.astype(float), k), rtol=1e-12)


def test_knn_selections_keep_the_most_isolated():
    payoff = _payoff(120, seed=10, low=-1000, high=1000)  # no ties
    cp = cupy()
    nov = ref_novelty._knn_novelty_distances(payoff.astype(float), 5)
    sco = ref_score._knn_score_distances(payoff.astype(float), 5)
    got = knn_novelty_selection(cp.asarray(payoff), 30, 5, DeviceRng(0, 0)).get()
    assert got.tolist() == np.sort(np.argsort(-nov)[:30]).tolist()
    got = knn_score_selection(cp.asarray(payoff), 30, 5, DeviceRng(0, 0)).get()
    assert got.tolist() == np.sort(np.argsort(-sco)[:30]).tolist()


# ── row-sum based ────────────────────────────────────────────────────────

def test_best_worst_random_groups():
    payoff = _payoff(100, seed=11, low=-1000, high=1000)
    scores = payoff.sum(axis=1)
    ranked = np.argsort(-scores)
    got = set(best_worst_random_selection(cupy().asarray(payoff), 20, 10, 15, DeviceRng(0, 0)).get().tolist())
    assert len(got) == 45
    assert set(ranked[:20]) <= got and set(ranked[-10:]) <= got
    assert len(got & set(ranked[20:-10])) == 15
    assert best_worst_random_selection(cupy().asarray(payoff), 60, 30, 20, DeviceRng(0, 0)).get().tolist() \
        == list(range(100))


def test_random_groups_are_uniform():
    payoff = cupy().asarray(np.zeros((40, 40), dtype=np.int32))
    hits = np.zeros(40)
    for gen in range(400):
        hits[best_worst_random_selection(payoff, 0, 0, 10, DeviceRng(5, gen)).get()] += 1
    assert np.all(np.abs(hits - 100) < 40)


def test_mixture_matches_the_reference_without_ties():
    payoff = _payoff(150, seed=12, low=-1000, high=1000)
    got = mixture_selection(cupy().asarray(payoff), 30, 60, 15, 10, 5, DeviceRng(0, 0)).get()
    want = ref_mixture.mixture_selection(payoff, 30, 60, 15, 10, 5)
    assert got.tolist() == want.tolist()
    with pytest.raises(ValueError):
        mixture_selection(cupy().asarray(payoff), 1, 2, 3, 0, 5, DeviceRng(0, 0))


def test_caps():
    payoff = _payoff(50, seed=13, low=-1000, high=1000)
    cp = cupy()
    top = SELECTION_STEPS["cap_top"](CapStepConfig(max_pop=10))(cp.asarray(payoff), _pop(50), DeviceRng(0, 0))
    assert top.get().tolist() == np.sort(np.argsort(-payoff.sum(axis=1))[:10]).tolist()
    rnd = SELECTION_STEPS["cap_random"](CapStepConfig(kind="cap_random", max_pop=10))(
        cp.asarray(payoff), _pop(50), DeviceRng(0, 0)).get()
    assert len(set(rnd.tolist())) == 10


# ── genome steps, nash, pipeline ─────────────────────────────────────────

def test_dedupe_keeps_first_occurrences():
    random.seed(14)
    genomes = [gen_tree(random.randint(1, 20)) for _ in range(60)]
    genomes += [list(genomes[i]) for i in (3, 7, 7, 50)]
    random.shuffle(genomes)
    first = {}
    for i, g in enumerate(genomes):
        first.setdefault(tuple(g), i)
    got = SELECTION_STEPS["dedupe"](DedupeStepConfig())(None, _pop(genomes=genomes), DeviceRng(0, 0)).get()
    assert got.tolist() == sorted(first.values())


def test_max_length():
    genomes = [[1, 0] * k for k in range(10)]
    got = SELECTION_STEPS["max_length"](MaxLengthStepConfig(max_length=8))(None, _pop(genomes=genomes),
                                                                           DeviceRng(0, 0)).get()
    assert got.tolist() == [0, 1, 2, 3, 4]


def test_nash_warns_loudly(capsys):
    payoff = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=np.int32)  # rock-paper-scissors
    got = SELECTION_STEPS["nash"](NashStepConfig())(cupy().asarray(payoff), _pop(3), DeviceRng(0, 0)).get()
    assert got.tolist() == [0, 1, 2]
    assert "SLOW HOST PATH" in capsys.readouterr().err


def test_pre_selection_only_takes_payoff_free_steps():
    build_selection([MaxLengthStepConfig(max_length=5), DedupeStepConfig()], pre=True)
    with pytest.raises(ValueError, match="pre_selection"):
        build_selection([LexicaseStepConfig()], pre=True)


def test_every_step_runs_through_the_registry():
    payoff = cupy().asarray(_payoff(40, seed=15))
    pop = _pop(40)
    for cfg in [SkimStepConfig(), LexicaseStepConfig(n_accepted=10), KNNNoveltyStepConfig(n_accepted=10),
                KNNScoreStepConfig(n_accepted=10), BestWorstRandomStepConfig(n_best=5, n_worst=2, n_rand=3),
                MixtureStepConfig(n_best=5, n_good=10, n_good_novel=3, n_novelty=2), CapStepConfig(max_pop=10)]:
        (select,) = build_selection([cfg])
        idx = cupy().asarray(select(payoff, pop, DeviceRng(0, 0))).get()
        assert (np.diff(idx) > 0).all() and idx.min() >= 0 and idx.max() < 40, cfg.kind

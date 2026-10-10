import numpy as np
import pytest

from config import BestWorstRandomStepConfig
from selection.base import build_selection
from selection.best_worst_random import best_worst_random_selection


def ranked_payoff():
    strengths = np.array([3, 0, 6, 2, 7, 1, 5, 4])
    return strengths[:, None] - strengths[None, :]


def test_best_worst_and_random_groups_are_disjoint():
    payoff = ranked_payoff()
    extremes = {1, 2, 4, 5}  # two best and two worst
    random_sets = set()
    for seed in range(20):
        np.random.seed(seed)
        survivors = best_worst_random_selection(payoff, 2, 2, 2)
        assert len(survivors) == 6
        assert np.array_equal(survivors, np.unique(survivors))
        assert extremes <= set(survivors)
        random_sets.add(tuple(sorted(set(survivors) - extremes)))
    assert len(random_sets) > 1


@pytest.mark.parametrize(
    "counts,expected",
    [((2, 0, 0), [2, 4]), ((0, 2, 0), [1, 5]), ((2, 2, 0), [1, 2, 4, 5])],
)
def test_score_ranking(counts, expected):
    assert best_worst_random_selection(ranked_payoff(), *counts).tolist() == expected


def test_ranks_overall_score_rather_than_best_individual_matchup():
    payoff = np.array([[0, 10, -20], [-10, 0, 1], [20, -1, 0]])
    assert best_worst_random_selection(payoff, 1, 1, 0).tolist() == [0, 2]


@pytest.mark.parametrize("counts", [(3, 2, 1), (0, 0, 6)])
def test_ties_and_randoms_are_reproducible_and_without_replacement(counts):
    payoff = np.zeros((20, 20))
    np.random.seed(42)
    first = best_worst_random_selection(payoff, *counts)
    np.random.seed(42)
    assert np.array_equal(best_worst_random_selection(payoff, *counts), first)
    assert len(first) == len(np.unique(first)) == 6
    outcomes = set()
    for seed in range(10):
        np.random.seed(seed)
        outcomes.add(tuple(best_worst_random_selection(payoff, *counts)))
    assert len(outcomes) > 1


@pytest.mark.parametrize(
    "n,counts,expected",
    [
        (0, (2, 2, 2), []),
        (1, (0, 1, 0), [0]),
        (3, (0, 0, 0), []),
        (3, (2, 1, 0), [0, 1, 2]),
        (3, (9, 0, 0), [0, 1, 2]),
        (3, (0, 9, 0), [0, 1, 2]),
        (3, (0, 0, 9), [0, 1, 2]),
        (3, (2, 2, 2), [0, 1, 2]),
    ],
)
def test_population_boundaries(n, counts, expected):
    result = best_worst_random_selection(np.zeros((n, n)), *counts)
    assert result.tolist() == expected
    assert np.issubdtype(result.dtype, np.integer)


@pytest.mark.parametrize("counts", [(-1, 0, 0), (0, -1, 0), (0, 0, -1)])
def test_rejects_negative_counts(counts):
    with pytest.raises(ValueError, match="non-negative"):
        best_worst_random_selection(np.zeros((2, 2)), *counts)


@pytest.mark.parametrize("counts", [(1.5, 0, 0), (0, 1.5, 0), (0, 0, 1.5)])
def test_rejects_noninteger_counts(counts):
    with pytest.raises(TypeError):
        best_worst_random_selection(np.zeros((2, 2)), *counts)


@pytest.mark.parametrize(
    "payoff", [np.zeros(3), np.zeros((2, 3)), np.array([[np.nan]]), np.array([[np.inf]])]
)
def test_rejects_invalid_payoff(payoff):
    with pytest.raises(ValueError):
        best_worst_random_selection(payoff, 1, 0, 0)


def test_pipeline_registration_and_pre_selection_rejection():
    cfg = BestWorstRandomStepConfig(n_best=1, n_worst=1, n_rand=0)
    select, = build_selection([cfg])
    assert select(ranked_payoff(), [[i] for i in range(8)]).tolist() == [1, 4]
    with pytest.raises(ValueError, match="pre_selection"):
        build_selection([cfg], pre=True)

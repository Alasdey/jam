import numpy as np
import pytest

from config import KNNNoveltyStepConfig
from selection.base import build_selection
from selection.knn_novelty import knn_novelty_selection


def test_isolated_loser_beats_clustered_winners():
    payoff = np.array([[0, 0, -1], [0, 0, -1], [1, 1, 0]])
    # Reversing reward signs preserves distances, even though the isolated
    # individual now loses against both opponents.
    for matrix in (payoff, -payoff):
        assert knn_novelty_selection(matrix, 1, 1).tolist() == [2]


@pytest.mark.parametrize("k", [1, 3, 999])
def test_matches_direct_euclidean_reference_across_blocks(k):
    payoff = np.random.default_rng(12).integers(-1, 2, (260, 23), dtype=np.int8)
    distances = np.linalg.norm(
        payoff.astype(float)[:, None, :] - payoff[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    scores = np.sort(distances, axis=1)[:, :min(k, 259)].mean(axis=1)
    np.random.seed(42)
    shuffled = np.random.permutation(len(payoff))
    expected = np.sort(shuffled[np.argsort(-scores[shuffled], kind="stable")][:17])
    np.random.seed(42)
    assert np.array_equal(knn_novelty_selection(payoff, 17, k), expected)


def test_ties_are_random_and_reproducible():
    np.random.seed(7)
    expected = np.sort(np.random.permutation(20)[:5])
    np.random.seed(7)
    actual = knn_novelty_selection(np.zeros((20, 20)), 5, 3)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("n,count,expected", [(0, 3, []), (1, 1, [0]), (3, 5, [0, 1, 2]), (3, 0, [])])
def test_population_boundaries(n, count, expected):
    assert knn_novelty_selection(np.zeros((n, n)), count, 10).tolist() == expected


@pytest.mark.parametrize("payoff,count,k", [(np.zeros(3), 1, 1), (np.zeros((2, 2)), -1, 1), (np.zeros((2, 2)), 1, 0), (np.array([[np.nan]]), 1, 1)])
def test_invalid_inputs(payoff, count, k):
    with pytest.raises(ValueError):
        knn_novelty_selection(payoff, count, k)


def test_pipeline_registration():
    cfg = KNNNoveltyStepConfig(n_accepted=1, k=1)
    select, = build_selection([cfg])
    payoff = np.array([[0, 0, -1], [0, 0, -1], [1, 1, 0]])
    assert select(payoff, [[1], [2], [3]]).tolist() == [2]
    with pytest.raises(ValueError, match="pre_selection"):
        build_selection([cfg], pre=True)

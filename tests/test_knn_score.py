import numpy as np
import pytest

from config import KNNScoreStepConfig
from selection.base import build_selection
from selection.knn_score import knn_score_selection


def clustered_scores_payoff():
    # Scores are [0, 0, 1, 3]. The first pair has equal scores despite
    # different reward profiles; the last pair is closest in reward space.
    return np.array([[0, 1, -1, 0], [-1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0]])


def test_selects_smallest_distances_using_only_scalar_scores():
    payoff = clustered_scores_payoff()
    for matrix in (payoff, -payoff):
        assert knn_score_selection(matrix, 2, 1).tolist() == [0, 1]


def test_self_play_does_not_contribute_to_score():
    payoff = clustered_scores_payoff()
    # Including these diagonal rewards would select the final two programs.
    np.fill_diagonal(payoff, [1, -1, 1, -1])
    assert knn_score_selection(payoff, 2, 1).tolist() == [0, 1]


@pytest.mark.parametrize("k", [1, 7, 999])
def test_matches_scalar_reference_with_int8_scores_across_blocks(k):
    rng = np.random.default_rng(12)
    payoff = np.where(
        rng.random((260, 260)) < np.linspace(0.01, 0.99, 260)[:, None], 1, -1
    ).astype(np.int8)
    # Widen before summing: scalar scores and their distances exceed int8.
    widened = payoff.astype(float)
    coordinates = widened.sum(axis=1) - np.diag(widened)
    assert coordinates.min() < -128 and coordinates.max() > 127
    distances = np.abs(coordinates[:, None] - coordinates[None, :])
    np.fill_diagonal(distances, np.inf)
    density = np.sort(distances, axis=1)[:, :min(k, len(payoff) - 1)].mean(axis=1)
    np.random.seed(42)
    shuffled = np.random.permutation(len(payoff))
    expected = np.sort(shuffled[np.argsort(density[shuffled], kind="stable")][:37])
    np.random.seed(42)
    assert np.array_equal(knn_score_selection(payoff, 37, k), expected)


def test_identical_scores_are_distinct_neighbors_and_ties_are_reproducible():
    np.random.seed(7)
    expected = np.sort(np.random.permutation(20)[:5])
    np.random.seed(7)
    actual = knn_score_selection(np.zeros((20, 20)), 5, 3)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "n,count,expected",
    [(0, 3, []), (1, 1, [0]), (1, 0, []), (3, 5, [0, 1, 2]), (3, 0, [])],
)
def test_population_boundaries(n, count, expected):
    actual = knn_score_selection(np.zeros((n, n)), count, 10)
    assert actual.tolist() == expected
    assert np.issubdtype(actual.dtype, np.integer)


@pytest.mark.parametrize(
    "payoff,count,k",
    [
        (np.zeros(3), 1, 1),
        (np.zeros((2, 3)), 1, 1),
        (np.zeros((2, 2)), -1, 1),
        (np.zeros((2, 2)), 1, 0),
        (np.zeros((2, 2)), 1, -2),
        (np.array([[np.nan]]), 1, 1),
        (np.array([[np.inf]]), 1, 1),
    ],
)
def test_invalid_inputs(payoff, count, k):
    with pytest.raises(ValueError):
        knn_score_selection(payoff, count, k)


@pytest.mark.parametrize("count,k", [(1.5, 1), (1, 1.5)])
def test_counts_must_be_integers(count, k):
    with pytest.raises(TypeError):
        knn_score_selection(np.zeros((2, 2)), count, k)


def test_pipeline_registration_and_pre_selection_rejection():
    cfg = KNNScoreStepConfig(n_accepted=2, k=1)
    select, = build_selection([cfg])
    assert select(clustered_scores_payoff(), [[1], [2], [3], [4]]).tolist() == [0, 1]
    with pytest.raises(ValueError, match="pre_selection"):
        build_selection([cfg], pre=True)

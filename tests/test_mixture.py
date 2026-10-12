import numpy as np
import pytest

from config import MixtureStepConfig
from selection.base import build_selection
from selection.mixture import mixture_selection


def ranked_payoff():
    scores = np.array([100, 99, 70, 65, 64, 10, -100])
    payoff = np.zeros((7, 7))
    payoff[0, 1] = scores[0]
    payoff[1:, 0] = scores[1:]
    # Make the second-best scalar score very novel in payoff-row space.
    # These cancelling rewards preserve its scalar score and self-play reward.
    payoff[1, 2:4] = [500, -500]
    return payoff


def test_good_pool_uses_scalar_quality_and_full_population_scalar_neighbors():
    # Best is 0; the next four by scalar payoff are 1, 2, 3, 4. Full-population
    # nearest score distances in that pool are 1, 5, 1, 1, so 2 wins. Restricting
    # neighbors to the pool would instead pick 1 (distance 29), while searching
    # outside the pool would pick 6 (distance 110).
    assert mixture_selection(ranked_payoff(), 1, 4, 1, 0, 1).tolist() == [0, 2]


def test_final_novelty_can_select_a_candidate_passed_over_in_good_pool():
    # Candidate 1 was passed over for scalar novelty but has the most isolated
    # payoff row. It is still eligible, and all three picks remain disjoint.
    assert mixture_selection(ranked_payoff(), 1, 4, 1, 1, 1).tolist() == [0, 1, 2]


def test_final_novelty_keeps_selected_programs_as_neighbors():
    payoff = np.zeros((6, 6))
    payoff[:, 0] = [100, 99, 98, 0, 0, 0]
    payoff[:, 1] = [0, 0, 0, -10, 0, 11]
    payoff[:, 2] = -payoff[:, 1]
    # After keeping 0 and 1, candidate 2 is still only distance 1 from 1.
    # Candidate 5 has the largest full-population nearest-row distance.
    # Restricting the neighbors to survivors left over would instead pick 2.
    assert mixture_selection(payoff, 2, 0, 0, 1, 1).tolist() == [0, 1, 5]


@pytest.mark.parametrize("k", [1, 7, 999])
def test_matches_direct_full_population_reference(k):
    payoff = np.random.default_rng(12).integers(-10, 11, (25, 25), dtype=np.int8)
    widened = payoff.astype(float)
    quality = widened.sum(axis=1)
    coordinates = quality - widened.diagonal()
    scalar_distances = np.abs(coordinates[:, None] - coordinates[None, :])
    row_distances = np.linalg.norm(widened[:, None, :] - widened[None, :, :], axis=2)
    np.fill_diagonal(scalar_distances, np.inf)
    np.fill_diagonal(row_distances, np.inf)
    neighbor_count = min(k, len(payoff) - 1)
    scalar_novelty = np.sort(scalar_distances, axis=1)[:, :neighbor_count].mean(axis=1)
    row_novelty = np.sort(row_distances, axis=1)[:, :neighbor_count].mean(axis=1)

    np.random.seed(42)
    tie_order = np.random.permutation(len(payoff)).tolist()
    ranked = sorted(tie_order, key=lambda i: -quality[i])
    best = ranked[:3]
    good_pool = set(ranked[3:13])
    good = sorted(
        (i for i in tie_order if i in good_pool), key=lambda i: -scalar_novelty[i]
    )[:4]
    picked = set(best + good)
    novel = sorted(
        (i for i in tie_order if i not in picked), key=lambda i: -row_novelty[i]
    )[:5]
    expected = sorted(best + good + novel)

    np.random.seed(42)
    actual = mixture_selection(payoff, 3, 10, 4, 5, k)
    assert actual.tolist() == expected


def test_ties_are_random_reproducible_and_without_replacement():
    payoff = np.zeros((20, 20))
    outcomes = set()
    for seed in range(10):
        np.random.seed(seed)
        first = mixture_selection(payoff, 3, 8, 2, 4)
        np.random.seed(seed)
        assert np.array_equal(mixture_selection(payoff, 3, 8, 2, 4), first)
        assert len(first) == 9
        assert np.array_equal(first, np.unique(first))
        outcomes.add(tuple(first))
    assert len(outcomes) > 1


@pytest.mark.parametrize(
    "n,counts,expected",
    [
        (0, (2, 4, 2, 1), []),
        (1, (0, 1, 1, 0), [0]),
        (3, (0, 3, 0, 0), []),
        (3, (9, 0, 0, 0), [0, 1, 2]),
        (3, (0, 9, 9, 0), [0, 1, 2]),
        (3, (0, 0, 0, 9), [0, 1, 2]),
        (3, (1, 2, 1, 1), [0, 1, 2]),
    ],
)
def test_population_boundaries(n, counts, expected):
    result = mixture_selection(np.zeros((n, n)), *counts)
    assert result.tolist() == expected
    assert np.issubdtype(result.dtype, np.integer)


def test_oversized_good_pool_is_clamped_without_changing_survivor_quota():
    payoff = ranked_payoff()
    survivors = mixture_selection(payoff, 1, 999, 1, 1, 1)
    assert survivors.tolist() == [0, 1, 6]


@pytest.mark.parametrize(
    "counts",
    [(-1, 0, 0, 0), (0, -1, 0, 0), (0, 1, -1, 0), (0, 0, 0, -1)],
)
def test_rejects_negative_counts(counts):
    with pytest.raises(ValueError, match="non-negative"):
        mixture_selection(np.zeros((2, 2)), *counts)


def test_rejects_good_novel_quota_larger_than_good_pool():
    with pytest.raises(ValueError, match="n_good_novel"):
        mixture_selection(np.zeros((2, 2)), 0, 1, 2, 0)


@pytest.mark.parametrize("position", range(5))
def test_counts_and_k_must_be_integers(position):
    arguments = [0, 1, 1, 0, 1]
    arguments[position] = 1.5
    with pytest.raises(TypeError):
        mixture_selection(np.zeros((2, 2)), *arguments)


@pytest.mark.parametrize("k", [0, -1])
def test_rejects_nonpositive_k(k):
    with pytest.raises(ValueError, match="positive"):
        mixture_selection(np.zeros((2, 2)), 0, 0, 0, 0, k)


@pytest.mark.parametrize(
    "payoff", [np.zeros(3), np.zeros((2, 3)), np.array([[np.nan]]), np.array([[np.inf]])]
)
def test_rejects_invalid_payoff(payoff):
    with pytest.raises(ValueError):
        mixture_selection(payoff, 0, 0, 0, 0)


def test_pipeline_registration_and_pre_selection_rejection():
    cfg = MixtureStepConfig(n_best=1, n_good=4, n_good_novel=1, n_novelty=1, k=1)
    select, = build_selection([cfg])
    assert select(ranked_payoff(), [[i] for i in range(7)]).tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="pre_selection"):
        build_selection([cfg], pre=True)

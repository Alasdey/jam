import numpy as np
import pytest

from config import LexicaseStepConfig
from selection.base import build_selection
from selection.lexicase import lexicase_selection


def test_case_order_selects_specialists_over_best_total(monkeypatch):
    payoff = np.array([[10, 0], [0, 10], [6, 6]])
    for order, expected in [([0, 1], 0), ([1, 0], 1)]:
        monkeypatch.setattr(np.random, "permutation", lambda n: np.array(order))
        assert lexicase_selection(payoff, 1).tolist() == [expected]


def test_later_cases_filter_only_remaining_candidates(monkeypatch):
    monkeypatch.setattr(np.random, "permutation", lambda n: np.arange(n))
    payoff = np.array([[10, 1], [10, 2], [0, 100]])
    assert lexicase_selection(payoff, 1).tolist() == [1]


def test_each_pick_reshuffles_and_excludes_previous_winners(monkeypatch):
    orders = iter([np.array([0, 1]), np.array([1, 0])])
    monkeypatch.setattr(np.random, "permutation", lambda n: next(orders))
    payoff = np.array([[10, 0], [0, 10], [6, 6]])
    assert lexicase_selection(payoff, 2).tolist() == [0, 1]


def test_ties_are_broken_among_survivors(monkeypatch):
    monkeypatch.setattr(np.random, "choice", lambda candidates: candidates[-1])
    assert lexicase_selection(np.zeros((4, 4)), 2).tolist() == [2, 3]


@pytest.mark.parametrize("shape,count,expected", [
    ((0, 0), 3, []), ((1, 1), 1, [0]), ((3, 3), 5, [0, 1, 2]),
    ((3, 3), 0, []),
])
def test_population_boundaries(shape, count, expected):
    result = lexicase_selection(np.zeros(shape), count)
    assert result.tolist() == expected
    assert result.dtype == np.dtype(int)


def test_no_cases_selects_unique_survivors():
    assert len(np.unique(lexicase_selection(np.empty((5, 0)), 3))) == 3


@pytest.mark.parametrize("payoff,count", [
    (np.zeros(3), 1), (np.zeros((2, 2)), -1), (np.array([[np.nan]]), 1),
])
def test_invalid_inputs(payoff, count):
    with pytest.raises(ValueError):
        lexicase_selection(payoff, count)


def test_registered_pipeline_step():
    step, = build_selection([LexicaseStepConfig(n_accepted=1)])
    payoff = np.array([[0, 1], [-1, 0]])
    assert step(payoff, [[1], [2]]).tolist() == [0]
    with pytest.raises(ValueError, match="lexicase"):
        build_selection([LexicaseStepConfig()], pre=True)


def _reference_lexicase(payoff, n_accepted):
    """Original selection loop, independent of both production paths."""
    remaining = np.arange(payoff.shape[0])
    if len(remaining) <= n_accepted:
        return remaining
    selected = []
    for _ in range(n_accepted):
        candidates = remaining
        for case in np.random.permutation(payoff.shape[1]):
            if len(candidates) == 1:
                break
            scores = payoff[candidates, case]
            candidates = candidates[scores == scores.max()]
        winner = np.random.choice(candidates)
        selected.append(winner)
        remaining = remaining[remaining != winner]
    return np.sort(np.asarray(selected, dtype=int))


def _assert_matches_reference(payoff, count, seed):
    initial_state = np.random.get_state()
    original_payoff = payoff.copy()
    try:
        np.random.seed(seed)
        expected = _reference_lexicase(payoff, count)
        expected_state = np.random.get_state()
        np.random.seed(seed)
        actual = lexicase_selection(payoff, count)
        actual_state = np.random.get_state()
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.dtype(int)
        assert actual_state[0] == expected_state[0]
        np.testing.assert_array_equal(actual_state[1], expected_state[1])
        assert actual_state[2:] == expected_state[2:]
        np.testing.assert_array_equal(payoff, original_payoff)
    finally:
        np.random.set_state(initial_state)


@pytest.mark.parametrize("seed", [0, 42, 999])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("pattern", ["duplicates", "continuous", "ties", "no_cases"])
def test_matches_original_survivors_and_rng(seed, layout, pattern):
    rng = np.random.default_rng(123)
    if pattern == "duplicates":
        # Unequal group sizes; selecting all but one also exhausts groups.
        payoff = np.repeat(rng.integers(-1, 2, size=(5, 13)), [1, 2, 3, 8, 16], axis=0)
    elif pattern == "continuous":
        payoff = rng.normal(size=(30, 13))
    elif pattern == "ties":
        payoff = np.zeros((30, 13), dtype=np.int32)
    else:
        payoff = np.empty((30, 0))
    if layout == "F":
        payoff = np.asfortranarray(payoff)
    elif layout == "strided":
        payoff = payoff[::-1, ::-1]
    for count in (0, 1, 15, 29, 30, 31):
        _assert_matches_reference(payoff, count, seed)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.bool_, ">i4"])
def test_numerically_equal_rows_and_numeric_dtypes(dtype):
    payoff = np.array([
        [0., -0., 1.], [-0., 0., 1.], [1., 0., 0.], [0., 0., 0.],
    ], dtype=dtype)
    for seed in range(10):
        _assert_matches_reference(payoff, 3, seed)


def test_large_integer_scores_remain_exact():
    high = np.iinfo(np.int64).max
    payoff = np.array([[high, 0], [high - 1, 1], [high, 0], [-high, 2]])
    for seed in range(10):
        _assert_matches_reference(payoff, 3, seed)


@pytest.mark.parametrize("levels,uses_fallback", [(8, False), (9, True)])
def test_fallback_cutoff_and_rng(monkeypatch, levels, uses_fallback):
    import selection.lexicase as module

    fallback = module._lexicase_array
    calls = []

    def record_fallback(payoff, count):
        calls.append(True)
        return fallback(payoff, count)

    monkeypatch.setattr(module, "_lexicase_array", record_fallback)
    # The many-valued case comes after one case has already been preprocessed.
    payoff = np.column_stack([np.zeros(18), np.arange(18) % levels])
    _assert_matches_reference(payoff, 15, 42)
    assert bool(calls) is uses_fallback

import numpy as np

from config import SkimStepConfig
from selection.base import build_skim
from selection.skim import (
    iterated_elimination_strictly_dominated_rows,
    iterated_elimination_strictly_dominated_rows_fast,
)

# Rock-paper-scissors: no strategy is dominated.
RPS = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0],
])

# A beats B, B beats C, A ties C. Antisymmetric (self-play payoff).
# Full-column semantics (slow): only C is dominated (by A), then A vs B
# differ on column C, so {A, B} survive.
# Symmetric IESDS (fast): after C's row AND column are removed, B becomes
# dominated by A, so only {A} survives.
CASCADE = np.array([
    [0, 1, 0],
    [-1, 0, 1],
    [0, -1, 0],
])


def test_fast_keeps_all_when_nothing_dominated():
    assert iterated_elimination_strictly_dominated_rows_fast(RPS).tolist() == [0, 1, 2]


def test_slow_keeps_all_when_nothing_dominated():
    assert iterated_elimination_strictly_dominated_rows(RPS).tolist() == [0, 1, 2]


def test_fast_removes_dominated_row():
    payoff = np.array([
        [0, 1],
        [-1, 0],
    ])
    assert iterated_elimination_strictly_dominated_rows_fast(payoff).tolist() == [0]


def test_fast_symmetric_iesds_cascades_through_removed_columns():
    assert iterated_elimination_strictly_dominated_rows_fast(CASCADE).tolist() == [0]


def test_slow_full_column_semantics_do_not_cascade():
    # Documents the intended divergence: slow compares over ALL columns,
    # fast removes eliminated strategies' columns too (symmetric IESDS).
    assert iterated_elimination_strictly_dominated_rows(CASCADE).tolist() == [0, 1]


def test_build_skim_step_returns_surviving_indices():
    step = build_skim(SkimStepConfig(n_rounds=1, fraction=1.0, n_accepted=None))
    assert step(CASCADE.copy()).tolist() == [0]


def test_build_skim_fraction_keeps_a_share_of_dominated():
    # A dominates everyone; B, C, D all get eliminated at fraction 1.0.
    payoff = np.array([
        [0, 1, 1, 1],
        [-1, 0, 1, 1],
        [-1, -1, 0, 0],
        [-1, -1, 0, 0],
    ])
    np.random.seed(0)
    step = build_skim(SkimStepConfig(n_rounds=1, fraction=0.5, n_accepted=None))
    kept = step(payoff).tolist()
    # survivor [0] plus int(3 * (1 - 0.5)) = 1 randomly retained dominated row
    assert len(kept) == 2
    assert 0 in kept


def test_build_skim_n_accepted_stops_early():
    step = build_skim(SkimStepConfig(n_rounds=5, fraction=1.0, n_accepted=100))
    # population already below n_accepted: one round runs, then it stops
    assert step(RPS.copy()).tolist() == [0, 1, 2]

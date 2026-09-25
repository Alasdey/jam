import numpy as np
import pytest

from config import CapStepConfig, DedupeStepConfig, ExperimentConfig, SkimStepConfig
from creation import CREATORS, build_creator
from creation.treemo import TreemoCreator
from interpreters import INTERPRETERS, build_interpreter
from rewards.base import REWARDS, build_reward
from selection.base import SELECTION_STEPS, build_selection


def test_unknown_interpreter_fails_loudly():
    cfg = ExperimentConfig(interpreter="does_not_exist")
    with pytest.raises(KeyError):
        build_interpreter(cfg)
    with pytest.raises(KeyError):
        build_creator(cfg)


def test_unknown_reward_fails_loudly():
    cfg = ExperimentConfig(reward="does_not_exist")
    with pytest.raises(KeyError):
        build_reward(cfg)


def test_creator_and_interpreter_registries_cover_same_keys():
    assert set(CREATORS) == set(INTERPRETERS)


def test_treemo_py_uses_tree_creator():
    creator = build_creator(ExperimentConfig(interpreter="treemo_py"))
    assert isinstance(creator, TreemoCreator)


def test_reward_specs_are_named_consistently():
    for name, spec in REWARDS.items():
        assert spec.name == name


def test_build_selection_composes_steps_in_order():
    steps = build_selection([SkimStepConfig(), CapStepConfig(max_pop=5)])
    assert len(steps) == 2
    payoff = np.zeros((3, 3), dtype=int)
    genomes = [[1], [2], [3]]
    for step in steps:
        idx = step(payoff, genomes)
        assert idx.tolist() == [0, 1, 2]  # nothing dominated, under the cap


def test_unknown_selection_step_fails_loudly():
    with pytest.raises(KeyError):
        SELECTION_STEPS["does_not_exist"]


def test_cap_top_keeps_best_row_sums():
    step = SELECTION_STEPS["cap_top"](CapStepConfig(max_pop=2))
    payoff = np.array([
        [0, -1, -1],
        [1, 0, 1],
        [1, -1, 0],
    ])
    assert step(payoff, [[1], [2], [3]]).tolist() == [1, 2]


def test_dedupe_keeps_first_occurrence_of_each_genome():
    step = SELECTION_STEPS["dedupe"](DedupeStepConfig())
    genomes = [[1, 2], [3], [1, 2], [3], [1, 2, 3]]
    payoff = np.zeros((5, 5), dtype=int)
    assert step(payoff, genomes).tolist() == [0, 1, 4]


def test_dedupe_is_exact_match_not_payoff_match():
    # every payoff row is identical here (an all-draws reward), so a step that
    # keyed on payoff rows would collapse the population to one individual
    step = SELECTION_STEPS["dedupe"](DedupeStepConfig())
    genomes = [[1], [2], [3]]
    payoff = np.zeros((3, 3), dtype=int)
    assert step(payoff, genomes).tolist() == [0, 1, 2]
    # ... and genomes that differ only in order are distinct programs
    assert step(payoff, [[1, 2], [2, 1], [1, 2]]).tolist() == [0, 1]


def test_dedupe_handles_empty_population():
    step = SELECTION_STEPS["dedupe"](DedupeStepConfig())
    idx = step(np.zeros((0, 0), dtype=int), [])
    assert idx.tolist() == []
    assert idx.dtype == np.dtype(int)


def test_dedupe_works_without_a_payoff_matrix():
    # the pre-evaluation pipeline calls steps with payoff=None
    step = SELECTION_STEPS["dedupe"](DedupeStepConfig())
    assert step(None, [[1], [1], [2]]).tolist() == [0, 2]


def test_pre_selection_rejects_steps_that_read_the_payoff():
    assert build_selection([DedupeStepConfig()], pre=True)
    with pytest.raises(ValueError, match="pre_selection"):
        build_selection([DedupeStepConfig(), SkimStepConfig()], pre=True)
    with pytest.raises(ValueError, match="cap_top"):
        build_selection([CapStepConfig()], pre=True)

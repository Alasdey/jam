import pytest

from config import ExperimentConfig
from creation import CREATORS, build_creator
from creation.treemo import TreemoCreator
from interpreters import INTERPRETERS, build_interpreter
from rewards.base import REWARDS, build_reward
from selection.base import SELECTION_STEPS


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


def test_unknown_selection_step_fails_loudly():
    with pytest.raises(KeyError):
        SELECTION_STEPS["does_not_exist"]

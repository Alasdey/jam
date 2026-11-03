import pytest

from config import ExperimentConfig
from interpreters import build_runtime
from rewards.base import REWARDS, build_reward
from selection.base import SELECTION_STEPS


def test_only_treemo_runs():
    with pytest.raises(ValueError, match="Treemo only"):
        build_runtime(ExperimentConfig(interpreter="subleq"))


def test_unknown_reward_fails_loudly():
    cfg = ExperimentConfig(reward="does_not_exist")
    with pytest.raises(KeyError):
        build_reward(cfg)


def test_reward_specs_are_named_consistently():
    for name, spec in REWARDS.items():
        assert spec.name == name


def test_unknown_selection_step_fails_loudly():
    with pytest.raises(KeyError):
        SELECTION_STEPS["does_not_exist"]

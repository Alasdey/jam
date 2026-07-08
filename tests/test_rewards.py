import random

import pytest

from creation.treemo import gen_tree
from interpreters.treemo_py.treemo import TreemoInterpreter
from rewards.blind_reward import reward as blind_reward
from rewards.placeholder_reward import reward as placeholder_reward
from rewards.quine_pressure_reward import reward as quine_pressure_reward


@pytest.fixture
def interp():
    return TreemoInterpreter(max_step=50, pass_mode=False, first_mode=False)


def _random_programs(n, seed):
    random.seed(seed)
    return [gen_tree(random.randint(5, 40)) for _ in range(n)]


def test_blind_reward_is_zero(interp):
    a, b = _random_programs(2, seed=3)
    assert blind_reward(interp, a, b) == 0


def test_placeholder_reward_antisymmetric(interp):
    progs = _random_programs(8, seed=5)
    for a in progs:
        for b in progs:
            assert placeholder_reward(interp, a, b) == -placeholder_reward(interp, b, a)


def test_quine_pressure_antisymmetric(interp):
    progs = _random_programs(8, seed=7)
    for a in progs:
        for b in progs:
            assert quine_pressure_reward(interp, a, b) == -quine_pressure_reward(interp, b, a)


def test_quine_pressure_self_play_is_draw(interp):
    for a in _random_programs(5, seed=9):
        assert quine_pressure_reward(interp, a, a) == 0

import random

import numpy as np
import pytest

from core.matchups import HostMatchups, block, play_one
from creation.treemo import gen_tree
from interpreters.treemo_py.treemo import TreemoInterpreter
from rewards.base import REWARDS

blind_reward = REWARDS["blind"].fn
placeholder_reward = REWARDS["placeholder"].fn
quine_pressure_reward = REWARDS["quine_pressure"].fn


@pytest.fixture
def interp():
    return TreemoInterpreter(max_step=50, pass_mode=False, first_mode=False)


def _random_programs(n, seed):
    random.seed(seed)
    return [gen_tree(random.randint(5, 40)) for _ in range(n)]


def _dp_lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def _similarity(output, reference):
    return _dp_lcs(output, reference) / len(reference) if output and reference else 0.0


# Each reward written one matchup at a time, as the rewards were defined
# before they became batched: the batched versions must agree exactly.
SCALAR = {
    "blind": lambda it, a, b: 0 if it.run(b, a)[0] == it.run(a, b)[0] else 1,
    "placeholder": lambda it, a, b: int(np.sign(
        len(it.run(a, [0, 1, 2, 3, 4, 5])[0]) - len(it.run(b, [0, 1, 2, 3, 4, 5])[0]))),
    "quine_pressure": lambda it, a, b: int(np.sign(
        _similarity(it.run(a, b)[0], a) - _similarity(it.run(b, a)[0], b))),
}


@pytest.mark.parametrize("name", sorted(REWARDS))
def test_batched_rewards_match_their_scalar_definitions(interp, name):
    ref = _random_programs(6, seed=11) + [[], [1, 0]]
    pop = _random_programs(5, seed=13) + ref[:2]
    got = REWARDS[name].fn(HostMatchups(interp, *block(ref, pop)))
    expected = [SCALAR[name](interp, a, b) for a in ref for b in pop]
    assert np.asarray(got).tolist() == expected


def test_blind_reward_compares_outputs(interp):
    a, b = _random_programs(2, seed=3)
    assert play_one(blind_reward, interp, a, a) == 0
    differ = interp.run(b, a)[0] != interp.run(a, b)[0]
    assert play_one(blind_reward, interp, a, b) == int(differ)


def test_placeholder_reward_antisymmetric(interp):
    progs = _random_programs(8, seed=5)
    for a in progs:
        for b in progs:
            assert play_one(placeholder_reward, interp, a, b) == -play_one(placeholder_reward, interp, b, a)


def test_quine_pressure_antisymmetric(interp):
    progs = _random_programs(8, seed=7)
    for a in progs:
        for b in progs:
            assert play_one(quine_pressure_reward, interp, a, b) == -play_one(quine_pressure_reward, interp, b, a)


def test_quine_pressure_self_play_is_draw(interp):
    for a in _random_programs(5, seed=9):
        assert play_one(quine_pressure_reward, interp, a, a) == 0

import random

import pytest

from config import ExperimentConfig
from creation.genetics import (
    CODE_CROSSOVER_OPS,
    CODE_MUTATION_OPS,
    TREE_CROSSOVER_OPS,
    TREE_MUTATION_OPS,
    _is_balanced,
)
from creation.treemo import gen_tree


@pytest.fixture
def cfg():
    return ExperimentConfig()


# ── Integer-program operators ──────────────────────────────────────────────


@pytest.mark.parametrize("op_name", list(CODE_MUTATION_OPS))
def test_code_mutation_preserves_length_and_bounds(cfg, op_name):
    random.seed(11)
    op = CODE_MUTATION_OPS[op_name]
    code = [random.randint(cfg.code.min_val, cfg.code.max_val) for _ in range(50)]
    mutated = op(code, cfg, 1.0)
    assert len(mutated) == len(code)
    assert all(cfg.code.min_val <= v <= cfg.code.max_val for v in mutated)
    # rate=0 is the identity
    assert op(code, cfg, 0.0) == code


@pytest.mark.parametrize("op_name", list(CODE_CROSSOVER_OPS))
def test_code_crossover_produces_int_list(cfg, op_name):
    random.seed(13)
    op = CODE_CROSSOVER_OPS[op_name]
    a = [1] * 30
    b = [2] * 40
    child = op(a, b)
    assert isinstance(child, list)
    assert set(child) <= {1, 2}
    # empty parent falls back to a copy of the first parent
    assert op(a, []) == a
    assert op(a, []) is not a


def test_two_point_crossover_keeps_first_parent_length(cfg):
    random.seed(17)
    a = [1] * 30
    b = [2] * 40
    child = CODE_CROSSOVER_OPS["two_point"](a, b)
    assert len(child) == len(a)


# ── Tree operators ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("op_name", list(TREE_MUTATION_OPS))
def test_tree_mutation_keeps_dyck_word_valid(op_name):
    random.seed(19)
    op = TREE_MUTATION_OPS[op_name]
    for _ in range(50):
        tree = gen_tree(random.randint(2, 30))
        mutated = op(tree, 1.0)
        assert _is_balanced(mutated)


@pytest.mark.parametrize("op_name", list(TREE_CROSSOVER_OPS))
def test_tree_crossover_keeps_dyck_word_valid(op_name):
    random.seed(23)
    op = TREE_CROSSOVER_OPS[op_name]
    for _ in range(50):
        a = gen_tree(random.randint(2, 30))
        b = gen_tree(random.randint(2, 30))
        child = op(a, b)
        assert _is_balanced(child)

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


def test_tree_delete_includes_children_of_implicit_root(monkeypatch):
    from creation.genetics import _mutate_tree_once
    monkeypatch.setattr(random, "choice", lambda choices: choices[0])
    assert _mutate_tree_once([1, 0, 1, 0], "delete") == [1, 0]


def test_swap_with_childless_node_moves_branch(monkeypatch):
    from creation.genetics import _mutate_tree_once
    # Two root children: first has a leaf, second is childless.
    choices = iter([(0, 4), (4, 6), (1, 3), (5, 5)])
    def choose(options):
        selected = next(choices)
        assert selected in options
        return selected
    monkeypatch.setattr(random, "choice", choose)
    assert _mutate_tree_once([1, 1, 0, 0, 1, 0], "swap") == [1, 0, 1, 1, 0, 0]


def test_swap_excludes_ancestors_and_descendants(monkeypatch):
    from creation.genetics import _mutate_tree_once
    tree = [1, 1, 1, 0, 0, 0]
    monkeypatch.setattr(random, "choice", lambda options: (1, 5))
    assert _mutate_tree_once(tree, "swap") == tree


def test_insert_samples_empirical_subtree_sizes(monkeypatch):
    import creation.genetics as genetics
    choices = iter([(0, 4), 2, 3])
    def choose(options):
        selected = next(choices)
        assert selected in options
        if all(isinstance(x, int) for x in options) and len(options) == 3:
            assert sorted(options) == [1, 2, 3]
        return selected
    monkeypatch.setattr(random, "choice", choose)
    monkeypatch.setattr(genetics, "_random_branch", lambda size: [1, 0] * size)
    assert genetics._mutate_tree_once([1, 1, 0, 0], "insert") == [1, 1, 0, 1, 0, 1, 0, 0]


def test_regenerate_preserves_selected_subtree_size(monkeypatch):
    import creation.genetics as genetics
    monkeypatch.setattr(random, "choice", lambda options: (0, 6))
    sizes = []
    def branch(size):
        sizes.append(size)
        return [1, 1, 1, 0, 0, 0]
    monkeypatch.setattr(genetics, "_random_branch", branch)
    assert genetics._mutate_tree_once([1, 1, 0, 1, 0, 0], "regenerate") == [1, 1, 1, 0, 0, 0]
    assert sizes == [3]


def test_tree_event_budget_scales_with_original_node_count(monkeypatch):
    import creation.genetics as genetics
    events = []
    monkeypatch.setattr(random, "random", lambda: 0.04)
    monkeypatch.setattr(random, "choice", lambda options: options[0])
    def apply(tree, operation):
        events.append(operation)
        return tree + [1, 0]
    monkeypatch.setattr(genetics, "_mutate_tree_once", apply)
    tree = [1, 0] * 10
    assert len(genetics.mutate_tree_subtree(tree, 0.05)) == 40
    assert events == ["delete"] * 10
    events.clear()
    assert genetics.mutate_tree_subtree(tree, 0) == tree
    assert events == []


@pytest.mark.parametrize("operation", ["delete", "swap", "insert", "regenerate"])
def test_each_tree_operation_preserves_validity_and_parent(operation):
    from creation.genetics import _mutate_tree_once
    random.seed(125)
    for size in range(1, 60):
        tree = gen_tree(size)
        original = tree.copy()
        child = _mutate_tree_once(tree, operation)
        assert tree == original
        assert _is_balanced(child)
        if operation in ("swap", "regenerate"):
            assert len(child) == len(tree)


def test_subleq_rate_is_per_integer(cfg, monkeypatch):
    from creation.subleq import SubleqCreator
    cfg.genetics.mutation_rate = 0.05
    draws = iter([0.01, 0.5, 0.04, 0.9])
    monkeypatch.setattr(random, "random", lambda: next(draws))
    monkeypatch.setattr(random, "randint", lambda low, high: high)
    assert SubleqCreator(cfg).mutate([0, 0, 0, 0]) == [cfg.code.max_val, 0, cfg.code.max_val, 0]

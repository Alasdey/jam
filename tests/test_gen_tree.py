import random

from tests.reference.creation_cpu import _is_balanced, gen_tree


def test_gen_tree_single_node_is_empty_word():
    assert gen_tree(1) == []


def test_gen_tree_is_valid_dyck_word():
    random.seed(7)
    for n in range(2, 40):
        for _ in range(20):
            tree = gen_tree(n)
            assert len(tree) == 2 * (n - 1)
            assert set(tree) <= {0, 1}
            assert _is_balanced(tree)

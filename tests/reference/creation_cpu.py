"""Test reference: the CPU Treemo creation operators (creation/genetics.py and gen_tree before the GPU port)."""

import random
from typing import List

def gen_tree(n: int = 3) -> list[int]:
    """Uniform random plane tree with n nodes, as a Dyck word of length 2(n-1)."""
    if n == 1:
        return []

    # Step 1: Generate a random sequence of (n-1) U's and (n-1) D's
    steps = [1] * (n - 1) + [-1] * (n - 1)
    random.shuffle(steps)

    # Step 2: Find the cyclic rotation that gives a valid Dyck path.
    # By the cycle lemma, the unique valid rotation starts just after
    # the position of the overall minimum (last occurrence, to be safe).
    min_val, min_idx = 0, 0
    height = 0
    for i, s in enumerate(steps):
        height += s
        if height <= min_val:          # use <= to get the last minimum
            min_val = height
            min_idx = i

    # Dyck word by cycling to the found unique correct position
    dyck = steps[min_idx + 1:] + steps[:min_idx + 1]

    # Converting to the 0 1 format
    res = [1 if i==1 else 0 for i in dyck]

    return res


# ── Tree helpers ───────────────────────────────────────────────────────────────

def _is_balanced(s: List[int]) -> bool:
    depth = 0
    for ch in s:
        if ch == 1:
            depth += 1
        elif ch == 0:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _subtrees_at_depth(tree: List[int], target_depth: int) -> List[tuple]:
    """Return (start, end) spans of subtrees whose opening node is at target_depth."""
    spans = []
    depth = 0
    start = None
    for i, ch in enumerate(tree):
        if ch == 1:
            if depth == target_depth:
                start = i
            depth += 1
        elif ch == 0:
            depth -= 1
            if depth == target_depth and start is not None:
                spans.append((start, i + 1))
                start = None
    return spans


def _all_subtree_spans(tree: List[int]) -> List[tuple]:
    """Return (start, end) spans for every non-root subtree (depth > 1)."""
    spans: List[tuple] = []
    depth = 0
    stack: List[int] = []
    for i, ch in enumerate(tree):
        if ch == 1:
            stack.append(i)
            depth += 1
        elif ch == 0:
            if stack:
                s = stack.pop()
                if depth > 1:  # skip the root subtree (whole tree)
                    spans.append((s, i + 1))
            depth -= 1
    return spans


def _leaf_positions(tree: List[int]) -> List[int]:
    return [i for i in range(len(tree) - 1) if tree[i] == 1 and tree[i + 1] == 0]


def _tree_max_depth(tree: List[int]) -> int:
    d = md = 0
    for ch in tree:
        if ch == 1:
            d += 1
            if d > md:
                md = d
        elif ch == 0:
            d -= 1
    return md


# ── Tree operators ─────────────────────────────────────────────────────────────

def mutate_tree_leaf(tree: List[int], rate: float) -> List[int]:
    """Randomly expand or contract a leaf node."""
    if not tree or random.random() > rate:
        return tree
    leaves = _leaf_positions(tree)
    if not leaves:
        return tree
    idx = random.choice(leaves)
    if random.random() < 0.5 and len(tree) > 2:
        return tree[:idx] + tree[idx + 2:]              # contract: delete leaf [1,0]
    else:
        return tree[:idx] + [1, 1, 0, 0] + tree[idx + 2:]  # expand: [1,0] → [1,[1,0],0]


def _node_spans(tree: List[int]) -> List[tuple[int, int]]:
    """All explicit nodes; the Dyck word's root is implicit and excluded."""
    stack = []
    spans = []
    for i, token in enumerate(tree):
        if token == 1:
            stack.append(i)
        else:
            spans.append((stack.pop(), i + 1))
    return spans


def _child_branches(tree: List[int], node: tuple[int, int]) -> List[tuple[int, int]]:
    start, end = node
    return [(start + 1 + a, start + 1 + b)
            for a, b in _subtrees_at_depth(tree[start + 1:end - 1], 0)]


def _random_branch(size: int) -> List[int]:
    # gen_tree omits its implicit root; a branch needs that root explicitly encoded.
    return [1] + gen_tree(size) + [0]


def _mutate_tree_once(tree: List[int], operation: str) -> List[int]:
    spans = _node_spans(tree)
    if not spans:
        return tree.copy()
    node = random.choice(spans)
    start, end = node
    if operation == "delete":
        return tree[:start] + tree[end:]
    if operation == "insert":
        # Empirical subtree sizes, including the whole tree's implicit root.
        sizes = [(b - a) // 2 for a, b in spans] + [len(tree) // 2 + 1]
        branch = _random_branch(random.choice(sizes))
        children = _child_branches(tree, node)
        position = random.choice([a for a, _ in children] + [end - 1])
        return tree[:position] + branch + tree[position:]
    if operation == "regenerate":
        return tree[:start] + _random_branch((end - start) // 2) + tree[end:]
    if operation == "swap":
        eligible = [(a, b) for a, b in spans if b <= start or a >= end]
        if not eligible:
            return tree.copy()
        other = random.choice(eligible)
        left = random.choice(_child_branches(tree, node) or [(end - 1, end - 1)])
        right = random.choice(_child_branches(tree, other) or [(other[1] - 1, other[1] - 1)])
        (a, b), (c, d) = sorted([left, right])
        return tree[:a] + tree[c:d] + tree[b:c] + tree[a:b] + tree[d:]
    raise ValueError(f"Unknown tree mutation operation: {operation}")


def mutate_tree_subtree(tree: List[int], rate: float) -> List[int]:
    """Apply uniformly chosen delete/swap/insert/regenerate mutation events.

    One Bernoulli trial per original non-root node sets the event count.
    Each event samples nodes uniformly from the current tree. Newly inserted
    nodes can be targets but do not increase the event budget. Empty trees
    remain empty because their only node is the excluded implicit root.
    """
    if not 0 <= rate <= 1:
        raise ValueError("mutation rate must be between 0 and 1")
    n_events = sum(random.random() < rate for _ in range(len(tree) // 2))
    result = tree.copy()
    for _ in range(n_events):
        operation = random.choice(("delete", "swap", "insert", "regenerate"))
        result = _mutate_tree_once(result, operation)
    return result


def crossover_tree_depth1(tree_a: List[int], tree_b: List[int]) -> List[int]:
    """Swap one depth-1 child subtree from tree_b into tree_a."""
    spans_a = _subtrees_at_depth(tree_a, 1)
    spans_b = _subtrees_at_depth(tree_b, 1)
    if not spans_a or not spans_b:
        return tree_a
    sa = random.choice(spans_a)
    sb = random.choice(spans_b)
    return tree_a[:sa[0]] + tree_b[sb[0]:sb[1]] + tree_a[sa[1]:]


def crossover_tree_random_depth(tree_a: List[int], tree_b: List[int]) -> List[int]:
    """Swap a subtree from tree_b at a randomly chosen shared depth into tree_a."""
    d_a = _tree_max_depth(tree_a)
    d_b = _tree_max_depth(tree_b)
    if d_a < 1 or d_b < 1:
        return tree_a
    for _ in range(5):  # up to 5 attempts to find a non-empty common depth
        depth = random.randint(1, min(d_a, d_b))
        spans_a = _subtrees_at_depth(tree_a, depth)
        spans_b = _subtrees_at_depth(tree_b, depth)
        if spans_a and spans_b:
            sa = random.choice(spans_a)
            sb = random.choice(spans_b)
            return tree_a[:sa[0]] + tree_b[sb[0]:sb[1]] + tree_a[sa[1]:]
    return tree_a



def make_offspring(
    creator,
    survivors: list,
    n_offspring: int,
    gc,
    interp=None,
) -> tuple[list, list[str], list[list[int]]]:
    """
    Returns (offspring, methods, parent_indices) where parent_indices[i] is
    the list of indices into survivors used to create offspring[i].
    """
    if not survivors or n_offspring == 0:
        return [], [], []
    offspring = []
    methods = []
    parent_indices = []
    for _ in range(n_offspring):
        if random.random() < gc.crossover_prob and len(survivors) >= 2:
            idxs = random.sample(range(len(survivors)), 2)
            pa, pb = survivors[idxs[0]], survivors[idxs[1]]
            child = None
            if interp is not None and random.random() < gc.homoiconic_prob:
                child = creator.homoiconic(interp, pa, pb)
            if child is None:
                child = creator.crossover(pa, pb)
                methods.append("crossover")
            else:
                methods.append("homoiconic")
            parent_indices.append(idxs)
        else:
            idx = random.randrange(len(survivors))
            child = creator.mutate(survivors[idx])
            methods.append("mutate")
            parent_indices.append([idx])
        offspring.append(child)
    return offspring, methods, parent_indices


TREE_MUTATION_OPS = {"leaf": mutate_tree_leaf, "subtree": mutate_tree_subtree}
TREE_CROSSOVER_OPS = {"depth1": crossover_tree_depth1, "random_depth": crossover_tree_random_depth}

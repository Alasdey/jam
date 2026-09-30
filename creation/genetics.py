
import random
from typing import Callable, Dict, List

MutateFn = Callable[[List[int], float], List[int]]
CrossoverFn = Callable[[List[int], List[int]], List[int]]


# ── Integer-program operators ─────────────────────────────────────────────────

def mutate_code_uniform(code: List[int], cfg, rate: float) -> List[int]:
    """Randomly replace each gene with probability rate."""
    result = code.copy()
    for i in range(len(result)):
        if random.random() < rate:
            result[i] = random.randint(cfg.code.min_val, cfg.code.max_val)
    return result


def mutate_code_creep(code: List[int], cfg, rate: float, delta: int = 5) -> List[int]:
    """Nudge each gene by ±delta with probability rate, clamped to [min_val, max_val]."""
    result = code.copy()
    for i in range(len(result)):
        if random.random() < rate:
            result[i] = max(
                cfg.code.min_val,
                min(cfg.code.max_val, result[i] + random.randint(-delta, delta)),
            )
    return result


def crossover_code_single(code_a: List[int], code_b: List[int]) -> List[int]:
    """Single-point crossover; any list of ints is a valid program."""
    if not code_a or not code_b:
        return code_a.copy()
    cut_a = random.randint(0, len(code_a))
    cut_b = random.randint(0, len(code_b))
    return code_a[:cut_a] + code_b[cut_b:]


def crossover_code_two_point(code_a: List[int], code_b: List[int]) -> List[int]:
    """Two-point crossover on the min-length aligned segment; tail kept from code_a."""
    if not code_a or not code_b:
        return code_a.copy()
    min_len = min(len(code_a), len(code_b))
    if min_len < 2:
        return crossover_code_single(code_a, code_b)
    p1, p2 = sorted(random.sample(range(min_len + 1), 2))
    return code_a[:p1] + code_b[p1:p2] + code_a[p2:]


def crossover_code_uniform(code_a: List[int], code_b: List[int]) -> List[int]:
    """Uniform (gene-wise coin-flip) crossover; longer tail kept from code_a."""
    if not code_a or not code_b:
        return code_a.copy()
    min_len = min(len(code_a), len(code_b))
    child = [b if random.random() < 0.5 else a for a, b in zip(code_a[:min_len], code_b[:min_len])]
    return child + code_a[min_len:]


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
    # Local import avoids the creator/genetics module import cycle. gen_tree
    # omits its implicit root; a branch needs that root explicitly encoded.
    from creation.treemo import gen_tree
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


# ── Operator registries ────────────────────────────────────────────────────────

CODE_MUTATION_OPS: Dict[str, Callable] = {
    "uniform": mutate_code_uniform,
    "creep":   mutate_code_creep,
}

CODE_CROSSOVER_OPS: Dict[str, CrossoverFn] = {
    "single_point": crossover_code_single,
    "two_point":    crossover_code_two_point,
    "uniform":      crossover_code_uniform,
}

TREE_MUTATION_OPS: Dict[str, MutateFn] = {
    "leaf":    mutate_tree_leaf,
    "subtree": mutate_tree_subtree,
}

TREE_CROSSOVER_OPS: Dict[str, CrossoverFn] = {
    "depth1":       crossover_tree_depth1,
    "random_depth": crossover_tree_random_depth,
}


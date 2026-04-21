
import random
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from config import ExperimentConfig


# ── Type aliases ──────────────────────────────────────────────────────────────

Program = Any  # List[int] for all interpreters
MutateFn = Callable[[Program, float], Program]    # (prog, rate) -> prog
CrossoverFn = Callable[[Program, Program], Program]


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


def mutate_tree_subtree(tree: List[int], rate: float) -> List[int]:
    """Replace a random non-root subtree with an empty leaf [1,0]."""
    if not tree or random.random() > rate:
        return tree
    spans = _all_subtree_spans(tree)
    if not spans:
        return tree
    s, e = random.choice(spans)
    return tree[:s] + [1, 0] + tree[e:]


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


# ── Homoiconic crossover ───────────────────────────────────────────────────────

def homoiconic_crossover(interp, cfg: ExperimentConfig, code_a, code_b) -> Optional[Any]:
    """
    Run interpreter(code_a, code_b) and treat the output as a new program.

    Every language in this framework is homoiconic: their output domain equals
    their program domain (List[int] for all interpreters). Returns None if the
    output is empty or structurally invalid.
    """
    output, _ = interp.run(code_a, code_b)
    if not output:
        return None
    if cfg.interpreter == "treemo":
        return output if _is_balanced(output) else None
    return output  # List[int] is always a valid program


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


# ── Operator set ──────────────────────────────────────────────────────────────

class OperatorSet(NamedTuple):
    """Bundles mutation and crossover callables for a given interpreter type."""
    mutate: MutateFn        # (program, rate: float) -> program
    crossover: CrossoverFn  # (program_a, program_b) -> program


def build_operator_set(cfg: ExperimentConfig) -> OperatorSet:
    """Resolve operator names from cfg.genetics into bound callables."""
    gc = cfg.genetics

    def _lookup(registry: dict, key: str, label: str) -> Callable:
        op = registry.get(key)
        if op is None:
            raise ValueError(f"Unknown {label}: {key!r}. Available: {sorted(registry)}")
        return op

    if cfg.interpreter == "treemo":
        raw_mut = _lookup(TREE_MUTATION_OPS, gc.tree_mutation_op, "tree mutation op")
        raw_xo  = _lookup(TREE_CROSSOVER_OPS, gc.tree_crossover_op, "tree crossover op")
        return OperatorSet(mutate=raw_mut, crossover=raw_xo)
    else:
        raw_mut = _lookup(CODE_MUTATION_OPS, gc.code_mutation_op, "code mutation op")
        raw_xo  = _lookup(CODE_CROSSOVER_OPS, gc.code_crossover_op, "code crossover op")
        _cfg = cfg  # capture by value for lambda
        return OperatorSet(
            mutate=lambda code, rate, _m=raw_mut, _c=_cfg: _m(code, _c, rate),
            crossover=raw_xo,
        )


# ── Unified offspring generation ───────────────────────────────────────────────

def make_offspring(
    cfg: ExperimentConfig,
    survivors: list,
    n_offspring: int,
    ops: OperatorSet,
    interp=None,
) -> list:
    """
    Generate n_offspring from survivors using the provided OperatorSet.

    Probabilities (crossover_prob, homoiconic_prob, mutation_rate) are read
    from cfg.genetics. When interp is given, homoiconic_crossover is attempted
    first for a fraction of crossover operations, falling back to ops.crossover
    if the output is invalid.
    """
    if not survivors or n_offspring == 0:
        return []
    gc = cfg.genetics
    offspring = []
    for _ in range(n_offspring):
        if random.random() < gc.crossover_prob and len(survivors) >= 2:
            pa, pb = random.sample(survivors, 2)
            child = None
            if interp is not None and random.random() < gc.homoiconic_prob:
                child = homoiconic_crossover(interp, cfg, pa, pb)
            if child is None:
                child = ops.crossover(pa, pb)
        else:
            parent = random.choice(survivors)
            child = ops.mutate(parent, gc.mutation_rate)
        offspring.append(child)
    return offspring

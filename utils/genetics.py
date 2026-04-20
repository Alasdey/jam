
import random
from typing import List, Optional, Any

from config import ExperimentConfig


# ── Integer-program operators (subleq, iconfractran) ──────────────────────────

def mutate_code(code: List[int], cfg, rate: float = 0.05) -> List[int]:
    result = code.copy()
    for i in range(len(result)):
        if random.random() < rate:
            result[i] = random.randint(cfg.code.min_val, cfg.code.max_val)
    return result


def crossover_code(code_a: List[int], code_b: List[int]) -> List[int]:
    """Single-point crossover. Any list of ints is a valid program so no
    structural constraints apply."""
    if not code_a or not code_b:
        return code_a.copy()
    cut_a = random.randint(0, len(code_a))
    cut_b = random.randint(0, len(code_b))
    return code_a[:cut_a] + code_b[cut_b:]


# ── Tree operators (treemo) ────────────────────────────────────────────────────

def _is_balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _subtrees_at_depth(tree: str, target_depth: int) -> List[tuple]:
    """Return (start, end) index spans of subtrees whose opening '(' is at
    target_depth. Used so crossover operates on children of the root rather
    than the root itself."""
    spans = []
    depth = 0
    start = None
    for i, ch in enumerate(tree):
        if ch == '(':
            if depth == target_depth:
                start = i
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == target_depth and start is not None:
                spans.append((start, i + 1))
                start = None
    return spans


def _leaf_positions(tree: str) -> List[int]:
    return [i for i in range(len(tree) - 1) if tree[i] == '(' and tree[i + 1] == ')']


def mutate_tree(tree: str, rate: float = 0.1) -> str:
    if not tree or random.random() > rate:
        return tree
    leaves = _leaf_positions(tree)
    if not leaves:
        return tree
    idx = random.choice(leaves)
    if random.random() < 0.5 and len(tree) > 2:
        return tree[:idx] + tree[idx + 2:]           # contract: delete leaf ()
    else:
        return tree[:idx] + '(())' + tree[idx + 2:]  # expand: () → (())


def crossover_tree(tree_a: str, tree_b: str) -> str:
    """Swap one depth-1 child subtree from tree_b into tree_a.
    Operates at depth 1 (children of the root node) so the result is always
    a single well-formed parenthesised tree."""
    spans_a = _subtrees_at_depth(tree_a, 1)
    spans_b = _subtrees_at_depth(tree_b, 1)
    if not spans_a or not spans_b:
        return tree_a
    sa = random.choice(spans_a)
    sb = random.choice(spans_b)
    return tree_a[:sa[0]] + tree_b[sb[0]:sb[1]] + tree_a[sa[1]:]


# ── Homoiconic crossover ───────────────────────────────────────────────────────

def homoiconic_crossover(interp, cfg: ExperimentConfig, code_a, code_b) -> Optional[Any]:
    """
    Run interpreter(code_a, code_b) and treat the output as a new program.

    Every language in this framework is homoiconic: their output domain is
    the same as their program domain (lists of ints for subleq/iconfractran,
    balanced-paren strings for treemo). Returns None if the output is empty
    or, for treemo, structurally invalid.
    """
    output, _ = interp.run(code_a, code_b)
    if not output:
        return None
    if cfg.interpreter == "treemo":
        candidate = output if isinstance(output, str) else ''.join(output)
        return candidate if _is_balanced(candidate) else None
    return output  # List[int] is always a valid program


# ── Unified offspring generation ───────────────────────────────────────────────

def make_offspring(
    cfg: ExperimentConfig,
    survivors: list,
    n_offspring: int,
    interp=None,
    mutation_rate: float = 0.05,
    crossover_prob: float = 0.5,
    homoiconic_prob: float = 0.3,
) -> list:
    """
    Generate n_offspring from survivors.

    When interp is provided a fraction (homoiconic_prob) of crossover
    operations use homoiconic_crossover; the rest use structural crossover.
    Crossover and mutation are dispatched based on cfg.interpreter so that
    tree programs always stay structurally valid.
    """
    if not survivors or n_offspring == 0:
        return []
    is_tree = cfg.interpreter == "treemo"
    offspring = []
    for _ in range(n_offspring):
        if random.random() < crossover_prob and len(survivors) >= 2:
            pa, pb = random.sample(survivors, 2)
            child = None
            if interp is not None and random.random() < homoiconic_prob:
                child = homoiconic_crossover(interp, cfg, pa, pb)
            if child is None:
                child = crossover_tree(pa, pb) if is_tree else crossover_code(pa, pb)
        else:
            parent = random.choice(survivors)
            child = mutate_tree(parent, mutation_rate) if is_tree else mutate_code(parent, cfg, mutation_rate)
        offspring.append(child)
    return offspring


from typing import List, Tuple
from random import randint

def gen_tree(n: int=3) -> List[int]:
    res = [1]
    space = 0
    dept = space
    for _ in range(n-1):
        space = randint(1, space+1)
        res += [0] * (dept+1-space)
        res += [1]
        dept = space
    res += [0] * (dept+1)
    return res

def tree_to_parent_list(code: List[int]) -> List[int]:
    assert len(code) % 2 == 0
    parent = [-1]
    current = 0
    for i in range(len(code)):
        if code[i] == 1:
            parent.append(current)
            current = len(parent)-1
        else:
            current = parent[current]
            parent.append(current)
    return parent

def greedy_pairs(tree: List[int]) -> List[List[int]]:
    """
    """
    m = []
    r = []
    os = 0
    for node in range(max(tree)+1):
        if tree.count(node)==4:
            inds = [i for i, val in enumerate(tree) if val==node]
            m.append((os + inds[0]-1, os + inds[1]))
            r.append((os + inds[2]-1, os + inds[3]))
            tree = tree[0: inds[0]] + tree[inds[3]+1: len(tree)]
            os += inds[3]+1 - inds[0]
    return (m, r)

def extract(tree, inds):
    """
    """
    return tree[inds[0]:inds[1]]

def tree_to_rules(code: List[int]):
    tree = tree_to_parent_list(code)
    m, r = greedy_pairs(tree)
    res = [(extract(code, m[i]), extract(code, r[i])) for i in range(len(m))]
    return res

def _list_find(haystack: List[int], needle: List[int]) -> int:
    n = len(needle)
    if n == 0:
        return 0
    for i in range(len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1


def interpret(rules: List[tuple], inp: List[int], max_step: int = 5,
              pass_mode: bool = True, first_mode: bool = False) -> List[int]:
    """
    Execute Treemo rules on inp for at most max_step firings.

    pass_mode : True  → a rule fires at most once before advancing to another
                False → a rule fires until it no longer matches before advancing
    first_mode: True  → after a rule fires and the interpreter advances, restart
                        from rule 0
                False → after a rule fires and the interpreter advances, continue
                        to the next rule in sequence
    """
    res = list(inp)
    n = len(rules)
    if n == 0:
        return res

    i = 0
    no_fire_streak = 0   # consecutive rules visited with zero firings
    current_rule_fired = False  # did rule i fire at least once? (matters for pass_mode=False)
    step = 0

    while step < max_step:
        pat = list(rules[i][0])
        rep = list(rules[i][1])
        idx = _list_find(res, pat)

        if idx != -1:
            if pat == rep:          # identity rule → halt
                return res
            # Fire
            res = res[:idx] + rep + res[idx + len(pat):]
            step += 1
            no_fire_streak = 0
            current_rule_fired = True

            if pass_mode:
                # Advance after exactly one firing
                if first_mode:
                    i = 0
                else:
                    i = (i + 1) % n
                current_rule_fired = False

            # else: stay on rule i (exhaust mode)

        else:
            # No match for rule i
            if not current_rule_fired:
                # Rule never fired since last advance → count toward halt
                no_fire_streak += 1
                if no_fire_streak >= n:
                    break
                i = (i + 1) % n
            else:
                # pass_mode=False: rule was exhausted after firing at least once
                no_fire_streak = 0
                current_rule_fired = False
                if first_mode:
                    i = 0
                else:
                    i = (i + 1) % n

    return res


def tree_to_dot(code: List[int]) -> str:
    edges = []
    stack = []
    node_id = 0

    for elem in code:
        if elem == 1:
            current = node_id
            node_id += 1
            if stack:
                edges.append(f"    {stack[-1]} -- {current};")
            stack.append(current)
        else:
            stack.pop()

    dot = "graph Tree {\n"
    dot += "\n".join(edges)
    dot += "\n}"
    return dot


def treemo(code: List[int], inp: List[int], max_step: int = 5,
           pass_mode: bool = True, first_mode: bool = False) -> List[int]:
    rules = tree_to_rules(code)
    return interpret(rules, inp, max_step, pass_mode, first_mode)


class TreemoInterpreter:
    def __init__(self, max_step: int = 50,
                 pass_mode: bool = True, first_mode: bool = False):
        self.max_step = max_step
        self.pass_mode = pass_mode
        self.first_mode = first_mode

    def run(self, code: List[int], inp: List[int]) -> Tuple[List[int], List[int]]:
        result = treemo(code, inp, max_step=self.max_step,
                        pass_mode=self.pass_mode, first_mode=self.first_mode)
        return result, code

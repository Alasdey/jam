
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
    for node in range(max(tree)):
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


def interpret(rules: List[tuple], inp: List[int], max_step: int=5) -> List[int]:
    res = list(inp)
    something_happened = True
    step = 0
    while something_happened and step < max_step:
        step += 1
        something_happened = False
        for rule in rules:
            idx = _list_find(res, rule[0])
            if idx != -1:
                if rule[0] == rule[1]:
                    return res
                res = res[:idx] + list(rule[1]) + res[idx + len(rule[0]):]
                something_happened = True
                break
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


def treemo(code: List[int], inp: List[int], max_step: int = 5) -> List[int]:
    rules = tree_to_rules(code)
    res = interpret(rules, inp, max_step)
    return res


class TreemoInterpreter:
    def __init__(self, max_step: int = 50):
        self.max_step = max_step

    def run(self, code: List[int], inp: List[int]) -> Tuple[List[int], List[int]]:
        result = treemo(code, inp, max_step=self.max_step)
        return result, code

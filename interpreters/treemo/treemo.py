
from typing import List
from random import randint
from time import time

def gen_tree(n: int=3) -> str:
    """
    """
    res = '('
    space = 0
    dept = space
    for _ in range(n-1):
        space = randint(1, space+1)
        res += ')' * (dept+1-space) 
        res += '('
        dept = space
    res += ')' * (dept+1)
    return res

def tree_to_parent_list(code: List[str]) -> List[str]:
    """
    """
    # Condition on correctness of the code
    assert len(code) % 2 == 0 
    parent = [-1]
    current = 0
    for i in range(len(code)):
        if code[i] == '(':
            parent.append(current)
            current = len(parent)-1
        else:
            current = parent[current]
            parent.append(current)
    return parent

def greedy_pairs(tree: List[int]) -> List[List[str]]:
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

def tree_to_rules(code: List[str]):
    """
    Takes a list of 0 and 1 that represent a tree as an input
    and outputs the corresponding list of match and replace rules
    args:
        code: List[str]
    """
    tree = tree_to_parent_list(code)
    m, r = greedy_pairs(tree)
    res = [(extract(code, m[i]), extract(code, r[i])) for i in range(len(m))]
    return res

def interpret(rules: List[tuple[str]], input: str, max_step:int=5) -> str:
    """
    """
    res = input
    something_happened = True
    step = 0
    while something_happened and step < max_step:
        step += 1
        something_happened = False
        for rule in rules:
            if rule[0] in res:
                if rule[0] == rule[1]:
                    # print('stopping condition')
                    return res
                res = res.replace(rule[0], rule[1], 1)
                something_happened = True
                break
    # print("step reached", step)
    return res


def tree_to_dot(code: str) -> str:
    edges = []
    stack = []
    node_id = 0

    for char in code:
        if char == '(':
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


def treemo(code: List[str], inp: List[str], max_step: int=5) -> List[str]:
    """
    """
    rules = tree_to_rules(code)
    res = interpret(rules, inp, max_step)
    return res


t0 = time()

size = 50
# step = 1000
# npop = 1000
# nref = 100

# pop = [gen_tree(size) for _ in range(npop)]
# ref = [gen_tree(size) for _ in range(nref)]
# res = []
# score = []
# for a in pop:
#     temp = []
#     score.append(0)
#     for b in ref:
#         alead = ((treemo(a, treemo(a, b, step))==treemo(b, treemo(a, b, step))))
#         blead = ((treemo(b, treemo(b, a, step))==treemo(a, treemo(b, a, step))))
#         temp.append(alead!=blead)
#         if alead!=blead:
#             if blead == True:
#                 score[-1] += 1
#             else:
#                 score[-1] += -1
#     res.append(temp.count(True))


# z = res.count(0)
# m = res.count(nref)
# h = len(res) - z - m

# print(res)
# print(score)
# print(min(score), sum(score)/len(score), max(score))
# print(z, h, m)


a = gen_tree(10**3)

with open('temp.txt', 'w') as f:
    f.write(tree_to_dot(a))

print(time()-t0)

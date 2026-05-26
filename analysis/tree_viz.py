from typing import List

def graphviz_translate(tree: List[int]):
    """
    Transform a tree into a string.
    That string is an input for graphviz for that tree.
    """
    res = "digraph G {\n"
    pile = [0]
    for i in range(len(tree)):
        if tree[i] == 1:
            res += f"{pile[-1]} -> {i}\n"
            pile.append(i)
        else:
            pile.pop(-1)
    res += "}"
    return res

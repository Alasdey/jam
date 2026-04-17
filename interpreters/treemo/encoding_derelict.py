import math

def szudzik_pair(x, y):
    return (x * x + x + y) if x >= y else (y * y + x)

def szudzik_unpair(z):
    s = math.isqrt(z)
    t = z - s * s
    if t < s:
        return t, s
    else:
        return s, t - s

def tree_to_num(subtrees):
    code = 0 
    # Reverse list so the first item changes the LSBs, not MSBs (optional but common convention)
    # Using fold: pair(item, previous_result)
    for val in reversed(subtrees):
        # We generally add 1 to distinction between "end of list" (0) and "value 0"
        # Since Szudzik maps (0,0)->0, we need to ensure the accumulator isn't ambiguous.
        # A simple way is to map the list [a, b] -> pair(a, pair(b, 0) + 1) + 1
        code = szudzik_pair(val, code) + 1 
    return code

def num_to_tree(n):
    subtrees = []
    curr = n
    while curr > 0:
        curr -= 1 # Reverse the +1 offset
        val, tail = szudzik_unpair(curr)
        subtrees.append(val)
        curr = tail
    return subtrees

# Checks
t = tree_to_num([5, 2, 9])
print(f"Encoded: {t}")
print(f"Decoded: {num_to_tree(t)}")
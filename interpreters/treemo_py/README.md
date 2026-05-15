---

## Treemo Language Specification

### 1. Objects

All objects in Treemo are **finite, rooted, ordered, unlabelled trees**. A node has an ordered list of child subtrees. A leaf is a node with no children.

---

### 2. Programs

A **program** is a tree. It encodes an ordered list of rewrite rules.

---

### 3. Rule Extraction

Traverse the program tree in **DFS pre-order**. For each node `N`:

- If `N` has **exactly 2 children**: it is a rule-node. Record `(first_child, second_child)` as a rule. Do **not** recurse into the children.
- Otherwise: recurse into the children.

The result is an ordered list of `(match, replacement)` pairs, each a tree.

---

### 4. Execution

The state is a single tree. On each step, scan the rule list in order. For the first rule whose match tree occurs as a subtree of the state, replace the **leftmost** occurrence (in DFS pre-order) with the replacement tree. If the match equals the replacement, halt instead. Repeat until no rule fires or `max_step` is reached.

```
state ← input
for step in 1..max_step:
    for each rule (match, replace):
        t ← leftmost subtree of state equal to match
        if found:
            if match == replace: return state
            state ← state with t replaced by replace
            break
    else:
        return state
return state
```

---

### 5. Termination

- No rule fires → halt.
- Fired rule has match = replacement → halt.
- `max_step` reached → halt.

---

### 6. Notes on the Reference Implementation

Trees are serialized as Dyck words over `{0,1}` (`1`=open, `0`=close). Subtree occurrence and replacement reduce to substring search and substitution on the flat bit sequence. Any injective serialization with this property is a valid substrate.

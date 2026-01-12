from typing import List, Tuple, Optional

# Types:
# A Rule is (head_list, tail_list) where each is List[int]
Rule = Tuple[List[int], List[int]]
Program = List[Rule]


def find_subseq(haystack: List[int], needle: List[int]) -> Optional[int]:
    """Return the first index where needle occurs as a contiguous subsequence in haystack, else None."""
    n = len(needle)
    if n == 0:
        # Empty pattern would match everywhere -> non-terminating in most systems.
        # You can change this behavior if you want, but this is a safe default.
        return None
    if n > len(haystack):
        return None
    # Leftmost-first match
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return None


def replace_once(lst: List[int], start: int, old_len: int, replacement: List[int]) -> List[int]:
    """Return a new list with lst[start:start+old_len] replaced by replacement."""
    return lst[:start] + replacement + lst[start + old_len:]


def step(transformer: Program, state: Program) -> Optional[Program]:
    """
    Perform exactly one rewrite.
    Scan order for match targets is:
      rule0.head, rule0.tail, rule1.head, rule1.tail, ...
    For each transformer rule, find the first match in that scan order,
    replace leftmost occurrence, and return the new state.
    """
    # Build scan targets in the requested order
    # Each entry: (rule_index, which_side, list_ref)
    scan_targets = []
    for ri, (h_list, t_list) in enumerate(state):
        scan_targets.append((ri, "head", h_list))
        scan_targets.append((ri, "tail", t_list))

    for (H, T) in transformer:
        for ri, side, lst in scan_targets:
            pos = find_subseq(lst, H)
            if pos is None:
                continue

            new_lst = replace_once(lst, pos, len(H), T)

            old_h, old_t = state[ri]
            if side == "head":
                updated_rule = (new_lst, old_t)
            else:
                updated_rule = (old_h, new_lst)

            new_state = state.copy()
            new_state[ri] = updated_rule
            return new_state

    return None


def run(transformer: Program, input_program: Program, max_steps: Optional[int] = None) -> Program:
    """
    Run until no rules apply (or until max_steps, if provided, to prevent infinite loops).
    Returns the modified input program.
    """
    state = input_program
    steps = 0
    while True:
        nxt = step(transformer, state)
        if nxt is None:
            return state
        state = nxt
        steps += 1
        if max_steps is not None and steps >= max_steps:
            return state


# --- tiny example ---
if __name__ == "__main__":
    # transformer rules: [1,2] -> [9], then [9,3] -> [7,7]
    transformer = [
        ([1, 2], [9]),
        ([9, 3], [7, 7]),
    ]

    # input program is also a list of rules
    input_prog = [
        ([0, 1, 2, 3], [4, 5]),
        ([1, 2, 1],     [3, 1, 2]),
    ]

    out = run(transformer, input_prog)
    print(out)

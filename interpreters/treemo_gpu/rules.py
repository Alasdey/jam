"""
Treemo rule extraction on the device, for many programs at once.

A program is a forest of trees under an implicit root, written as a balanced
0/1 word (1 opens a node, 0 closes it). Its rules are its nodes with exactly
two children that have no two-child ancestor, the implicit root included, in
pre-order; a rule's pattern and replacement are its two child subtrees
(interpreters/treemo_py/treemo.py tree_to_rules is the reference). Patterns
are bit ranges of the program, so nothing is copied.

Every step is a vectorized CuPy operation over all the programs' bits:
depths by a segmented cumulative sum; matching open and close by a sort on
(program, level); children counted between a node's bits one level down.
"""

from dataclasses import dataclass

import numpy as np

from core.device import cupy, exclusive_cumsum, kernel, launch, warp_launch

_SOURCE = "interpreters/treemo_gpu/rules.cu"
_CHUNK_BITS = 1 << 24  # bits extracted per pass, bounding temporary memory


@dataclass
class RuleTable:
    """Rules of a set of programs (device arrays), aligned with those programs."""
    table: object   # int32 (R, 4): pattern bit, pattern length, replacement bit, replacement length
    ident: object   # uint8 (R,): pattern == replacement
    start: object   # int64 per program: its first row in table
    count: object   # int64 per program
    growth: object  # int64 per program: max(0, largest replacement - pattern length)

    @staticmethod
    def concat(parts: list["RuleTable"]) -> "RuleTable":
        cp = cupy()
        shifts = np.cumsum([0] + [len(p.table) for p in parts[:-1]])
        return RuleTable(
            cp.concatenate([p.table for p in parts]).reshape(-1, 4),
            cp.concatenate([p.ident for p in parts]),
            cp.concatenate([p.start + int(s) for p, s in zip(parts, shifts)]),
            cp.concatenate([p.count for p in parts]),
            cp.concatenate([p.growth for p in parts]),
        )


def extract_rules(heap, offset, length) -> RuleTable:
    """Rules of the programs at word offsets `offset` with `length` bits in `heap`."""
    cp = cupy()
    offset = cp.asarray(offset, dtype=cp.int64)
    length = cp.asarray(length, dtype=cp.int32)
    lens = length.astype(cp.int64)
    if not len(lens):
        return _empty()
    # Chunks of whole programs of about _CHUNK_BITS bits.
    ends = cp.cumsum(lens)
    total = int(ends[-1])
    cuts = [0]
    if total > _CHUNK_BITS:
        marks = cp.arange(_CHUNK_BITS, total, _CHUNK_BITS, dtype=cp.int64)
        cuts += sorted(set(cp.asnumpy(cp.searchsorted(ends, marks, side="right")).tolist()) - {0})
    cuts.append(len(lens))
    parts = [_extract(heap, offset[a:b], length[a:b]) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    return parts[0] if len(parts) == 1 else RuleTable.concat(parts)


def _empty() -> RuleTable:
    cp = cupy()
    z = cp.zeros(0, dtype=cp.int64)
    return RuleTable(cp.zeros((0, 4), dtype=cp.int32), cp.zeros(0, dtype=cp.uint8), z, z.copy(), z.copy())


def _extract(heap, offset, length) -> RuleTable:
    cp = cupy()
    n_progs = len(offset)
    lens = length.astype(cp.int64)
    starts = exclusive_cumsum(lens)
    total = int(lens.sum())
    count = cp.zeros(n_progs, dtype=cp.int64)
    if total == 0:
        return RuleTable(cp.zeros((0, 4), dtype=cp.int32), cp.zeros(0, dtype=cp.uint8),
                         cp.zeros(n_progs, dtype=cp.int64), count, count.copy())

    bits = cp.empty(total, dtype=cp.uint8)
    warp_launch(kernel(_SOURCE, "unpack_tapes"), n_progs, heap, offset, length, starts, bits, np.int32(n_progs))
    index = cp.arange(total, dtype=cp.int64)
    gid = cp.searchsorted(starts, index, side="right") - 1  # program of each bit
    step = 2 * bits.astype(cp.int64) - 1
    running = cp.cumsum(step)
    before = cp.concatenate([cp.zeros(1, dtype=cp.int64), running])[starts]
    depth = running - before[gid]  # depth after each bit, within its program
    ends = starts + lens - 1
    nonempty = lens > 0
    unbalanced = bool((depth < 0).any()) or bool((depth[ends[nonempty]] != 0).any())
    if unbalanced:
        raise ValueError("a Treemo program must be a balanced 0/1 word (1 opens a node, 0 closes it)")

    pos = index - starts[gid]
    level = cp.minimum(depth - step, depth)  # opens: depth before; closes: depth after
    n_levels = int(level.max()) + 2
    group = gid * n_levels + level
    # Within one program and level, opens and closes alternate in bit order,
    # so sorting by (group, bit) pairs every open with its close.
    order = cp.lexsort(cp.stack([index, group]))
    match = cp.empty(total, dtype=cp.int64)
    match[order[0::2]] = order[1::2]
    match[order[1::2]] = order[0::2]

    # A node's children are the subtrees one level down between its open and
    # close: two bits of that level per child.
    span = int(lens.max()) + 1
    key = group[order] * span + pos[order]
    is_open = bits == 1
    o = cp.flatnonzero(is_open)
    lo = cp.searchsorted(key, (group[o] + 1) * span + pos[o])
    hi = cp.searchsorted(key, (group[o] + 1) * span + pos[match[o]])
    two = o[hi - lo == 4]
    # The implicit root's children are the top-level trees.
    root_children = cp.bincount(gid[is_open & (level == 0)], minlength=n_progs)

    # Rules are two-child nodes with no two-child ancestor; when the root has
    # two children, the root is the only rule.
    cover = cp.zeros(total + 1, dtype=cp.int64)
    if len(two):
        cp.add.at(cover, two + 1, 1)
        cp.add.at(cover, match[two], -1)
    inside = cp.cumsum(cover)[:-1]
    nodes = two[(inside[two] == 0) & (root_children[gid[two]] != 2)]
    c1 = nodes + 1
    c2 = match[c1] + 1
    table = cp.stack([pos[c1], pos[match[c1]] + 1 - pos[c1], pos[c2], pos[match[nodes]] - pos[c2]], axis=1)
    prog = gid[nodes]

    rooted = cp.flatnonzero(root_children == 2)
    if len(rooted):
        first = starts[rooted]
        plen = match[first] - first + 1
        root_table = cp.stack([cp.zeros_like(plen), plen, plen, lens[rooted] - plen], axis=1)
        table = cp.concatenate([table, root_table])
        prog = cp.concatenate([prog, rooted])
        order = cp.lexsort(cp.stack([table[:, 0], prog]))  # by program, then pre-order
        table, prog = table[order], prog[order]
    table = table.astype(cp.int32)

    count = cp.bincount(prog, minlength=n_progs).astype(cp.int64)
    growth = cp.zeros(n_progs, dtype=cp.int64)
    ident = cp.zeros(len(table), dtype=cp.uint8)
    if len(table):
        cp.maximum.at(growth, prog, (table[:, 3] - table[:, 1]).astype(cp.int64))
        base = 32 * offset[prog]
        launch(kernel(_SOURCE, "rule_ident"), len(table), heap, table, base, ident, np.int32(len(table)))
    return RuleTable(table, ident, exclusive_cumsum(count), count, growth)

"""
Batched Treemo on CUDA: one warp per (program, input) execution.

Results are identical to interpreters/treemo_c: the kernel replays the C
interpreter's rule-advancement logic and scan watermarks. Rewards reach the
device through DeviceMatchups (interpreters/treemo_gpu/matchups.py); this
module holds the runtime they share: rule extraction, kernels, C fallback.

CuPy is imported on first device use, so importing this module needs no GPU.
Single executions (`run`) stay on C: a launch per tree would only add latency.
"""

from collections import OrderedDict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from core.matchups import zipped
from interpreters.treemo_c.treemo import TreemoInterpreter as _CTreemo

_KERNELS = ("treemo_run", "treemo_gather", "treemo_lcs", "treemo_lcs_small", "treemo_equal", "treemo_ones",
            "treemo_concat")
_RULE_CACHE_SIZE = 20_000  # genomes; a few generations of turnover


def extract_rules(codes: Sequence[bytes]) -> list:
    """
    Rule tables of many programs at once, vectorized over all their bits.

    For each program: None if it is not a balanced 0/1 Dyck word, else
    (rules, ident) with rules an (nr, 4) int32 array of (pattern start,
    pattern length, replacement start, replacement length) in program bits,
    and ident the per-rule pattern == replacement flags, in the order of
    interpreters/treemo_py.tree_to_rules. A rule is every node with exactly
    two children that has no two-child ancestor (the implicit root included),
    in pre-order; its pattern and replacement are the two child subtrees.
    """
    n_codes = len(codes)
    out = [None] * n_codes
    if not n_codes:
        return out
    lens = np.fromiter((len(c) for c in codes), dtype=np.int64, count=n_codes)
    starts = np.zeros(n_codes, dtype=np.int64)
    np.cumsum(lens[:-1], out=starts[1:])
    nonempty = lens > 0
    bits = np.frombuffer(b"".join(codes), dtype=np.uint8).astype(np.int64)
    gid = np.repeat(np.arange(n_codes), lens)
    step = 2 * bits - 1
    depth = np.cumsum(step)
    depth -= np.concatenate(([0], depth))[starts][gid]  # depth after each bit, per program

    valid = np.bincount(gid, weights=(bits > 1) | (depth < 0), minlength=n_codes) == 0
    valid[nonempty] &= depth[(starts + lens - 1)[nonempty]] == 0
    for p in np.flatnonzero(valid & ~nonempty):
        out[p] = (np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.uint8))
    idx = np.flatnonzero(valid[gid])  # bits of valid programs, kept contiguous
    if not len(idx):
        return out

    g = gid[idx]
    pos = idx - starts[g]
    level = np.minimum(depth[idx] - step[idx], depth[idx])  # opens: before, closes: after
    # Within one program and level, opens and closes alternate in bit order, so a
    # stable sort by (program, level) pairs every open with its close.
    n_levels = int(level.max()) + 2
    group = g * n_levels + level
    order = np.argsort(group, kind="stable")
    match = np.empty(len(idx), dtype=np.int64)
    match[order[0::2]] = order[1::2]
    match[order[1::2]] = order[0::2]

    # A node's children are the subtrees one level down between its open and
    # close: two bits of that level per child.
    span = int(lens.max()) + 1
    key = group[order] * span + pos[order]
    is_open = bits[idx] == 1
    o = np.flatnonzero(is_open)
    lo = np.searchsorted(key, (group[o] + 1) * span + pos[o])
    hi = np.searchsorted(key, (group[o] + 1) * span + pos[match[o]])
    two = o[hi - lo == 4]
    # The implicit root's children are the top-level trees.
    root_children = np.bincount(g[is_open & (level == 0)], minlength=n_codes)

    # Rules are two-child nodes with no two-child ancestor.
    cover = np.zeros(len(idx) + 1, dtype=np.int64)
    np.add.at(cover, two + 1, 1)
    np.add.at(cover, match[two], -1)
    inside = np.cumsum(cover)[:-1]
    nodes = two[(inside[two] == 0) & (root_children[g[two]] != 2)]
    c1 = nodes + 1
    c2 = match[c1] + 1
    table = np.stack([pos[c1], pos[match[c1]] + 1 - pos[c1],
                      pos[c2], pos[match[nodes]] - pos[c2]], axis=1).astype(np.int32)
    bounds = np.searchsorted(g[nodes], np.arange(n_codes + 1))

    first_bit = np.searchsorted(idx, starts)
    for p in np.flatnonzero(valid & nonempty).tolist():
        if root_children[p] == 2:  # the root is the only rule: (first tree, second tree)
            plen = int(match[first_bit[p]] - first_bit[p]) + 1
            rules = np.array([[0, plen, plen, lens[p] - plen]], dtype=np.int32)
        else:
            rules = table[bounds[p]:bounds[p + 1]]
        code = codes[p]
        ident = np.fromiter(
            (pl == rl and code[ps:ps + pl] == code[rs:rs + rl] for ps, pl, rs, rl in rules.tolist()),
            dtype=np.uint8, count=len(rules),
        )
        out[p] = (rules, ident)
    return out


class TreemoInterpreter:
    """
    Treemo interpreter with batched CUDA execution.

    memory_mb bounds the device memory of one execution launch (states and
    scan watermarks); larger batches are split into several launches.
    """

    def __init__(self, max_step: int = 50, pass_mode: bool = True,
                 first_mode: bool = False, memory_mb: int = 1024):
        if memory_mb < 1:
            raise ValueError("memory_mb must be positive")
        self.max_step = max_step
        self.pass_mode = bool(pass_mode)
        self.first_mode = bool(first_mode)
        self.memory_mb = memory_mb
        self._c = _CTreemo(max_step, pass_mode, first_mode)
        self._rules: OrderedDict = OrderedDict()
        self._cp = None
        self._kernels: dict = {}
        self.last_stats: dict = {}  # DeviceMatchups.stats of the last batch call below

    def run(self, code: List[int], inp: List[int]) -> Tuple[List[int], List[int]]:
        return self._c.run(code, inp)

    def matchups(self, programs: Sequence, a_ids, b_ids):
        """A batch of matchups on the device (see core.matchups)."""
        from interpreters.treemo_gpu.matchups import DeviceMatchups

        # Batches differ in size, so cached blocks of earlier ones rarely fit
        # later ones: release them once they add up, keeping device memory
        # near one batch's worth.
        pool = self._cuda().get_default_memory_pool()
        if pool.total_bytes() - pool.used_bytes() > 2**30:
            pool.free_all_blocks()
        return DeviceMatchups(self, programs, a_ids, b_ids)

    # Batch conveniences over (codes[k], inputs[k]) pairs.

    def run_batch(self, codes: Sequence[List[int]],
                  inputs: Sequence[List[int]]) -> List[Tuple[List[int], List[int]]]:
        """(output, code) for each pair, like `run`, in order."""
        m = self.matchups(*zipped(codes, inputs))
        outs = m.run(m.a, m.b)
        result = [(list(out), code) for out, code in zip(m._sequences(outs.ids), codes)]
        self.last_stats = m.stats
        return result

    def lengths(self, codes: Sequence[List[int]], inputs: Sequence[List[int]]) -> np.ndarray:
        """Output length of each pair; outputs never leave the device."""
        m = self.matchups(*zipped(codes, inputs))
        result = m.run(m.a, m.b).length()
        self.last_stats = m.stats
        return result

    def self_similarities(self, codes: Sequence[List[int]],
                          inputs: Sequence[List[int]]) -> np.ndarray:
        """LCS(output, code) / len(code) per pair, 0 when either is empty (quine pressure)."""
        m = self.matchups(*zipped(codes, inputs))
        result = m.run(m.a, m.b).similarity(m.a)
        self.last_stats = m.stats
        return result

    def _rule_tables(self, codes: Sequence[bytes]) -> list:
        """Rule tables from an LRU cache, so survivors are extracted once per lifetime."""
        missing = [c for c in dict.fromkeys(codes) if c not in self._rules]
        for code, table in zip(missing, extract_rules(missing)):
            self._rules[code] = table
        tables = []
        for code in codes:
            self._rules.move_to_end(code)
            tables.append(self._rules[code])
        while len(self._rules) > _RULE_CACHE_SIZE:
            self._rules.popitem(last=False)
        return tables

    def _cuda(self):
        if self._cp is None:
            try:
                import cupy as cp
            except ImportError as exc:
                raise RuntimeError(
                    "payoff.backend='cuda' needs CuPy built for your CUDA driver "
                    "(cupy-cuda12x or cupy-cuda13x); see interpreters/treemo_gpu/README.md"
                ) from exc
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("payoff.backend='cuda' found no CUDA device")
            module = cp.RawModule(code=Path(__file__).with_name("kernels.cu").read_text(),
                                  options=("--std=c++17",))
            self._kernels = {name: module.get_function(name) for name in _KERNELS}
            self._cp = cp
        return self._cp

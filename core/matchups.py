"""
The batch of matchups a reward scores, on the GPU.

A reward is a function of one `Matchups` returning one value per matchup:

    def quine_pressure(m: Matchups):
        imprint_a = m.run(m.a, m.b).similarity(m.a)
        imprint_b = m.run(m.b, m.a).similarity(m.b)
        return np.sign(imprint_a - imprint_b)

Matchup k is player m.a[k] (whose reward it is) against m.b[k]. Sequences come
as Tapes, one sequence per matchup:

    m.a, m.b              the two players' programs
    m.const(seq)          the same 0/1 sequence in every matchup
    m.run(code, inp)      the output of code[k] run on inp[k]
    m.memory(code, inp)   the interpreter's second result (treemo: the code)
    t.concat(u)           t[k] followed by u[k]

and are measured into device (CuPy) arrays, one entry per matchup:

    t.length()   t.count(symbol)   t.equals(u)   t.lcs(u)   t.similarity(ref)

Combine those with NumPy or CuPy functions (np.sign, np.where, arithmetic all
work on CuPy arrays and keep them on the device) and return the array; the
payoff engine casts it to int32. Each distinct execution or measurement runs
once per batch: in a square block, run(m.b, m.a) reuses run(m.a, m.b).

m.map(fn, *args) runs any Python per matchup on host copies of the tapes. It
is the one way off the device, so it is slow and says so loudly every time.
"""

from typing import Callable

import numpy as np

from core.device import cupy, kernel, loud_warning, warp_launch, launch
from core.population import Population, pack
from interpreters.treemo_gpu.rules import RuleTable, extract_rules

_SOURCE = "interpreters/treemo_gpu/kernels.cu"
_LOW = (1 << 32) - 1
_THREADS = 128
_LCS_MAX_BITS = 16 * 32 * 32  # the warp LCS keeps the shorter side in registers
_THREAD_LCS_WORDS = (4, 8, 16, 24, 32, 48, 64, 96, 128)  # sizes of the one-job-per-thread LCS


def lcs_length(a, b) -> int:
    """Exact LCS length, bit-parallel over the shorter sequence (Crochemore et al. 2001)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return 0
    masks: dict = {}
    for i, symbol in enumerate(b):
        masks[symbol] = masks.get(symbol, 0) | (1 << i)
    full = (1 << len(b)) - 1
    v = full
    for symbol in a:
        m = masks.get(symbol, 0)
        v = ((v + (v & m)) | (v & ~m)) & full
        if not v:  # all of b matched: the rest of a cannot change the result
            break
    return len(b) - v.bit_count()


class TapeStore:
    """Tapes in one growing device heap (core/cuda/tapes.cuh layout), addressed by id."""

    def __init__(self, programs: Population, spare_words: int = 1 << 20):
        cp = cupy()
        self.heap = cp.zeros(len(programs.heap) + spare_words, dtype=cp.uint32)
        self.heap[:len(programs.heap)] = programs.heap
        self.used = len(programs.heap)
        self.offset = programs.offset.astype(cp.int64)
        self.length = programs.length.astype(cp.int64)

    def __len__(self) -> int:
        return len(self.offset)

    def _reserve(self, words: int) -> int:
        """Word offset of `words` fresh words; may move the heap."""
        cp = cupy()
        need = self.used + words
        if need > len(self.heap):
            grown = cp.zeros(max(need, 2 * len(self.heap)), dtype=cp.uint32)
            grown[:self.used] = self.heap[:self.used]
            self.heap = grown
        base, self.used = self.used, need
        return base

    def allocate(self, lengths, words):
        """New tape ids for tapes of `lengths` bits taking `words` words each, and their word offsets."""
        cp = cupy()
        from core.device import exclusive_cumsum
        dst = exclusive_cumsum(words)
        base = self._reserve(int(words.sum())) if len(words) else self.used
        dst += base
        ids = cp.arange(len(self), len(self) + len(lengths), dtype=cp.int64)
        self.offset = cp.concatenate([self.offset, dst])
        self.length = cp.concatenate([self.length, cp.asarray(lengths, dtype=cp.int64)])
        return ids, dst

    def upload(self, seqs) -> object:
        cp = cupy()
        words, offsets = pack(seqs)
        lengths = np.fromiter(map(len, seqs), dtype=np.int64, count=len(seqs))
        base = self._reserve(len(words) - 1)
        self.heap[base:base + len(words) - 1] = cp.asarray(words[1:])
        ids = cp.arange(len(self), len(self) + len(seqs), dtype=cp.int64)
        self.offset = cp.concatenate([self.offset, cp.asarray(offsets - 1 + base)])
        self.length = cp.concatenate([self.length, cp.asarray(lengths)])
        return ids

    def download(self, ids) -> list[list[int]]:
        cp = cupy()
        ids = cp.asarray(ids, dtype=cp.int64)
        heap = cp.asnumpy(self.heap[:self.used])
        from core.population import unpack
        return unpack(heap, cp.asnumpy(self.offset[ids]), cp.asnumpy(self.length[ids]))

    def bits(self, ids):
        """(bit offset, length as int32) of tapes, for the measurement kernels."""
        cp = cupy()
        return 32 * self.offset[ids], self.length[ids].astype(cp.int32)


class _Rules:
    """Rule rows of the store's tapes, extracted on first use as a program."""

    def __init__(self, n_tapes: int, known: RuleTable | None):
        cp = cupy()
        self.start = cp.full(n_tapes, -1, dtype=cp.int64)
        self.count = cp.zeros(n_tapes, dtype=cp.int64)
        self.growth = cp.zeros(n_tapes, dtype=cp.int64)
        self.table = cp.zeros((0, 4), dtype=cp.int32)
        self.ident = cp.zeros(0, dtype=cp.uint8)
        if known is not None:
            k = len(known.start)
            self.table, self.ident = known.table, known.ident
            self.start[:k], self.count[:k], self.growth[:k] = known.start, known.count, known.growth

    def ensure(self, store: TapeStore, codes) -> None:
        cp = cupy()
        if len(self.start) < len(store):
            grow = len(store) - len(self.start)
            self.start = cp.concatenate([self.start, cp.full(grow, -1, dtype=cp.int64)])
            self.count = cp.concatenate([self.count, cp.zeros(grow, dtype=cp.int64)])
            self.growth = cp.concatenate([self.growth, cp.zeros(grow, dtype=cp.int64)])
        missing = cp.unique(codes[self.start[codes] < 0])
        if len(missing):
            new = extract_rules(store.heap, store.offset[missing], store.length[missing])
            self.start[missing] = new.start + len(self.table)
            self.count[missing] = new.count
            self.growth[missing] = new.growth
            self.table = cp.concatenate([self.table, new.table]).reshape(-1, 4)
            self.ident = cp.concatenate([self.ident, new.ident])


class Tapes:
    """One sequence per matchup; see the module docstring."""

    __slots__ = ("m", "ids")

    def __init__(self, m: "Matchups", ids):
        self.m = m
        self.ids = ids

    def length(self):
        self.m._own(self)
        return self.m._store.length[self.ids]

    def count(self, symbol: int):
        return self.m._unary(("count", int(symbol)), self)

    def equals(self, other: "Tapes"):
        return self.m._binary("equals", self, other).astype(bool)

    def lcs(self, other: "Tapes"):
        return self.m._binary("lcs", self, other)

    def similarity(self, ref: "Tapes"):
        """LCS(self, ref) / len(ref): how much of ref appears in self; 0.0 when either is empty."""
        cp = cupy()
        lcs = self.lcs(ref).astype(cp.float64)
        n = ref.length()
        nz = (n > 0) & (self.length() > 0)
        return cp.where(nz, lcs / cp.maximum(n, 1), 0.0)

    def concat(self, other: "Tapes") -> "Tapes":
        return Tapes(self.m, self.m._binary("concat", self, other))

    def map(self, fn: Callable):
        return self.m.map(fn, self)


class Matchups:
    """
    A batch of matchups over a population: matchup k is programs[a[k]]
    against programs[b[k]]. `rules` may hand over the programs' rule table
    when the caller already has it (it is extracted on first use otherwise).
    """

    def __init__(self, runtime, programs: Population, a, b, rules: RuleTable | None = None):
        cp = cupy()
        self.runtime = runtime
        self._a = cp.asarray(a, dtype=cp.int64)
        self._b = cp.asarray(b, dtype=cp.int64)
        self.n = len(self._a)
        self._store = TapeStore(programs)
        self._rules = _Rules(len(self._store), rules)
        self._memo: dict = {}
        self.stats = {"runs": 0, "launches": 0, "host_values": 0}

    # Properties, not attributes: Tapes point at their batch, and a cycle would
    # keep a finished batch (and its device memory) alive until the next GC.
    @property
    def a(self) -> Tapes:
        return Tapes(self, self._a)

    @property
    def b(self) -> Tapes:
        return Tapes(self, self._b)

    def run(self, code: Tapes, inp: Tapes) -> Tapes:
        return Tapes(self, self._binary("run", code, inp))

    def memory(self, code: Tapes, inp: Tapes) -> Tapes:
        return Tapes(self, self._binary("memory", code, inp))

    def const(self, seq) -> Tapes:
        cp = cupy()
        seq = list(seq)
        if any(s not in (0, 1) for s in seq):
            raise ValueError("Treemo tapes hold only the symbols 0 and 1")
        tape = int(self._store.upload([seq])[0])
        return Tapes(self, cp.full(self.n, tape, dtype=cp.int64))

    def map(self, fn: Callable, *args):
        """fn(*values) per matchup on the host: Tapes become lists, arrays their entries. Slow."""
        cp = cupy()
        loud_warning(f"Matchups.map({getattr(fn, '__name__', fn)}) copies {self.n:,} matchups to the host "
                     "and runs Python on each one")
        columns = []
        for arg in args:
            if isinstance(arg, Tapes):
                self._own(arg)
                uniq, inverse = cp.unique(arg.ids, return_inverse=True)
                seqs = self._store.download(uniq)
                columns.append((True, [seqs[i] for i in cp.asnumpy(inverse).tolist()]))
            else:
                values = cp.asnumpy(arg) if isinstance(arg, cp.ndarray) else np.asarray(arg)
                if values.shape != (self.n,):
                    raise ValueError(f"map arguments need one entry per matchup, got shape {values.shape}")
                columns.append((False, values.tolist()))
        self.stats["host_values"] += self.n
        return cp.asarray([fn(*(list(column[k]) if tape else column[k] for tape, column in columns))
                           for k in range(self.n)])

    # ── deduplication and memo ───────────────────────────────────────────

    def _own(self, *tapes: Tapes) -> None:
        for t in tapes:
            if not isinstance(t, Tapes) or t.m is not self:
                raise ValueError("Tapes from a different batch of matchups")

    def _unary(self, op, x: Tapes):
        cp = cupy()
        self._own(x)
        uniq, inverse = cp.unique(x.ids, return_inverse=True)
        return self._memoized(op, uniq, lambda keys: self._compute(op, keys, None))[inverse]

    def _binary(self, op, x: Tapes, y: Tapes):
        cp = cupy()
        self._own(x, y)
        keys, inverse = cp.unique((x.ids << 32) | y.ids, return_inverse=True)
        values = self._memoized(op, keys, lambda k: self._compute(op, k >> 32, k & _LOW))
        return values[inverse]

    def _memoized(self, op, keys, compute):
        """Values for sorted distinct keys, computing only those not seen before."""
        cp = cupy()
        if op not in self._memo or not len(self._memo[op][0]):
            values = cp.asarray(compute(keys))
            self._memo[op] = (keys, values)
            return values
        known, known_values = self._memo[op]
        pos = cp.minimum(cp.searchsorted(known, keys), len(known) - 1)
        hit = known[pos] == keys
        values = cp.empty(len(keys), dtype=known_values.dtype)
        values[hit] = known_values[pos[hit]]
        miss = ~hit
        if bool(miss.any()):
            fresh = cp.asarray(compute(keys[miss])).astype(known_values.dtype)
            values[miss] = fresh
            merged = cp.concatenate([known, keys[miss]])
            order = cp.argsort(merged)
            self._memo[op] = (merged[order], cp.concatenate([known_values, fresh])[order])
        return values

    # ── operations on distinct tape ids ──────────────────────────────────

    def _compute(self, op, x, y):
        if op == "run":
            self._rules.ensure(self._store, x)
            self.stats["runs"] += len(x)
            return self.runtime.execute(self._store, self._rules, x, y)
        if op == "memory":
            return x.copy()  # treemo returns the program it ran
        if op == "concat":
            return self._concat(x, y)
        if op == "lcs":
            return self._lcs(x, y)
        if op == "equals":
            return self._pairwise("treemo_equal", x, y)
        if op[0] == "count":
            return self._count(x, op[1])
        raise ValueError(f"unknown operation {op!r}")

    def _launch(self, name: str, n: int, *args) -> None:
        warp_launch(kernel(_SOURCE, name), n, *args)
        self.stats["launches"] += 1

    def _pairwise(self, name: str, x, y):
        cp = cupy()
        out = cp.empty(len(x), dtype=cp.int32)
        self._launch(name, len(x), self._store.heap, *self._store.bits(x), *self._store.bits(y), out,
                     np.int32(len(x)))
        return out.astype(cp.int64)

    def _count(self, x, symbol: int):
        cp = cupy()
        if symbol not in (0, 1):
            return cp.zeros(len(x), dtype=cp.int64)  # tapes hold no other symbol
        ones = cp.empty(len(x), dtype=cp.int32)
        self._launch("treemo_ones", len(x), self._store.heap, *self._store.bits(x), ones, np.int32(len(x)))
        ones = ones.astype(cp.int64)
        return ones if symbol == 1 else self._store.length[x] - ones

    def _concat(self, x, y):
        lengths = self._store.length[x] + self._store.length[y]
        ids, dst = self._store.allocate(lengths, (lengths + 31) // 32 + 1)
        self._launch("treemo_concat", len(x), self._store.heap, *self._store.bits(x), *self._store.bits(y), dst,
                     np.int32(len(x)))
        return ids

    def _lcs(self, x, y):
        cp = cupy()
        out = cp.zeros(len(x), dtype=cp.int64)
        lx, ly = self._store.length[x], self._store.length[y]
        swap = lx < ly  # the shorter tape is the bit-vector side
        text = cp.where(swap, y, x)
        ref = cp.where(swap, x, y)
        short = cp.minimum(lx, ly)
        dev = short <= _LCS_MAX_BITS
        words = (short + 31) // 32
        bounds = cp.asarray(_THREAD_LCS_WORDS)
        bucket = cp.searchsorted(bounds, words)  # == len(bounds) past 128 words
        bucket = cp.where(dev, bucket, -1)
        for b, k in enumerate(_THREAD_LCS_WORDS):
            jobs = cp.flatnonzero(bucket == b)
            if len(jobs):
                d_out = cp.empty(len(jobs), dtype=cp.int32)
                launch(kernel(_SOURCE, "treemo_lcs_small"), len(jobs),
                       self._store.heap, *self._store.bits(text[jobs]), *self._store.bits(ref[jobs]), d_out,
                       np.int32(len(jobs)), np.int32(k), block=_THREADS)
                self.stats["launches"] += 1
                out[jobs] = d_out
        jobs = cp.flatnonzero(bucket == len(_THREAD_LCS_WORDS))
        if len(jobs):
            out[jobs] = self._pairwise("treemo_lcs", text[jobs], ref[jobs])
        host = cp.flatnonzero(~dev)
        if len(host):
            loud_warning(f"{len(host):,} LCS of two tapes both longer than {_LCS_MAX_BITS:,} bits "
                         "run in Python on the host")
            seqs_x = self._store.download(x[host])
            seqs_y = self._store.download(y[host])
            out[host] = cp.asarray([lcs_length(p, q) for p, q in zip(seqs_x, seqs_y)], dtype=cp.int64)
            self.stats["host_values"] += len(host)
        return out

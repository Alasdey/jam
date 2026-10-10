"""
The batch of matchups a reward scores, the same on every payoff backend.

A reward is a function of one `Matchups` returning one value per matchup:

    def quine_pressure(m: Matchups) -> np.ndarray:
        imprint_a = m.run(m.a, m.b).similarity(m.a)
        imprint_b = m.run(m.b, m.a).similarity(m.b)
        return np.sign(imprint_a - imprint_b)

Matchup k is player m.a[k] (whose reward it is) against m.b[k]. Sequences come
as Tapes, one sequence per matchup:

    m.a, m.b              the two players' programs
    m.const(seq)          the same sequence in every matchup
    m.run(code, inp)      the output of code[k] run on inp[k]
    m.memory(code, inp)   the interpreter's second result (treemo: the code)
    t.concat(u)           t[k] followed by u[k]

and are measured into numpy arrays, one entry per matchup:

    t.length()   t.count(symbol)   t.equals(u)   t.lcs(u)   t.similarity(ref)

Combine those with numpy (np.sign, np.where, ...) and return the array; the
payoff engine casts it to int. Anything else goes through m.map(fn, *args):
fn gets each matchup's sequences as lists (and the entries of array
arguments) and runs in plain Python, so it is fully general but slow on large
blocks. m.interpreter is the scalar interpreter, for map functions that need
run() themselves.

A backend computes each distinct execution or measurement once per batch: in
a square self-play block, run(m.b, m.a) reuses every output of run(m.a, m.b).
"""

from typing import Callable, Sequence

import numpy as np

_LOW = (1 << 32) - 1


def block(ref: Sequence, pop: Sequence):
    """
    The distinct programs of ref and pop (as tuples), and the a and b program
    of every matchup of the ref x pop block, in row-major order.
    """
    ids: dict = {}

    def index(population):
        return np.fromiter((ids.setdefault(tuple(code), len(ids)) for code in population),
                           dtype=np.int64, count=len(population))

    rows, cols = index(ref), index(pop)
    return list(ids), np.repeat(rows, len(pop)), np.tile(cols, len(ref))


def zipped(codes: Sequence, inputs: Sequence):
    """Like block(), for the pairs (codes[k], inputs[k]) instead of a full block."""
    if len(codes) != len(inputs):
        raise ValueError("codes and inputs must have the same length")
    ids: dict = {}
    a = np.fromiter((ids.setdefault(tuple(c), len(ids)) for c in codes), dtype=np.int64, count=len(codes))
    b = np.fromiter((ids.setdefault(tuple(i), len(ids)) for i in inputs), dtype=np.int64, count=len(inputs))
    return list(ids), a, b


def lcs_length(a: Sequence, b: Sequence) -> int:
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


class Tapes:
    """One sequence per matchup; see the module docstring."""

    __slots__ = ("m", "ids")

    def __init__(self, m: "Matchups", ids: np.ndarray):
        self.m = m
        self.ids = ids

    def length(self) -> np.ndarray:
        self.m._own(self)
        return self.m._lengths[self.ids]

    def count(self, symbol: int) -> np.ndarray:
        return self.m._unary(("count", int(symbol)), self)

    def equals(self, other: "Tapes") -> np.ndarray:
        return self.m._binary("equals", self, other).astype(bool)

    def lcs(self, other: "Tapes") -> np.ndarray:
        return self.m._binary("lcs", self, other)

    def similarity(self, ref: "Tapes") -> np.ndarray:
        """LCS(self, ref) / len(ref): how much of ref appears in self; 0.0 when either is empty."""
        lcs = self.lcs(ref)
        n = ref.length()
        out = np.zeros(len(lcs), dtype=np.float64)
        nz = (n > 0) & (self.length() > 0)
        out[nz] = lcs[nz] / n[nz]
        return out

    def concat(self, other: "Tapes") -> "Tapes":
        return Tapes(self.m, self.m._binary("concat", self, other))

    def map(self, fn: Callable) -> np.ndarray:
        return self.m.map(fn, self)


class Matchups:
    """
    A batch of matchups (see the module docstring). Backends hold the tapes
    and implement _store, _compute and _sequences on distinct tape ids; this
    class deduplicates per call and remembers results across calls.
    """

    def __init__(self, interpreter, a_ids: np.ndarray, b_ids: np.ndarray):
        self.interpreter = interpreter
        self.n = len(a_ids)
        self._lengths = np.zeros(0, dtype=np.int64)  # per tape id
        self._memo: dict = {}
        self._a = np.asarray(a_ids, dtype=np.int64)
        self._b = np.asarray(b_ids, dtype=np.int64)

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

    def const(self, seq: Sequence[int]) -> Tapes:
        tape = int(self._store([tuple(seq)])[0])
        return Tapes(self, np.full(self.n, tape, dtype=np.int64))

    def map(self, fn: Callable, *args) -> np.ndarray:
        """fn(*values) per matchup: Tapes become fresh lists, arrays their entries."""
        columns = []
        for arg in args:
            if isinstance(arg, Tapes):
                self._own(arg)
                uniq, inverse = np.unique(arg.ids, return_inverse=True)
                seqs = self._sequences(uniq)
                columns.append((True, [seqs[i] for i in inverse.tolist()]))
            else:
                values = np.asarray(arg)
                if values.shape != (self.n,):
                    raise ValueError(f"map arguments need one entry per matchup, got shape {values.shape}")
                columns.append((False, values.tolist()))
        return np.array([fn(*(list(column[k]) if tape else column[k] for tape, column in columns))
                         for k in range(self.n)])

    # ── deduplication and memo ───────────────────────────────────────────

    def _own(self, *tapes: Tapes) -> None:
        for t in tapes:
            if not isinstance(t, Tapes) or t.m is not self:
                raise ValueError("Tapes from a different batch of matchups")

    def _unary(self, op, x: Tapes) -> np.ndarray:
        self._own(x)
        uniq, inverse = np.unique(x.ids, return_inverse=True)
        return self._memoized(op, uniq, lambda keys: self._compute(op, keys, None))[inverse]

    def _binary(self, op, x: Tapes, y: Tapes) -> np.ndarray:
        self._own(x, y)
        keys, inverse = _distinct_pairs(x.ids, y.ids)
        values = self._memoized(op, keys, lambda k: self._compute(op, k >> 32, k & _LOW))
        return values[inverse]

    def _memoized(self, op, keys: np.ndarray, compute) -> np.ndarray:
        """Values for sorted distinct keys, computing only those not seen before."""
        if op not in self._memo:
            values = np.asarray(compute(keys))
            self._memo[op] = (keys, values)
            return values
        known, known_values = self._memo[op]
        pos = np.searchsorted(known, keys)
        hit = pos < len(known)
        hit[hit] = known[pos[hit]] == keys[hit]
        values = np.empty(len(keys), dtype=known_values.dtype)
        values[hit] = known_values[pos[hit]]
        if not hit.all():
            fresh = np.asarray(compute(keys[~hit]))
            values[~hit] = fresh
            merged = np.concatenate([known, keys[~hit]])
            order = np.argsort(merged, kind="stable")
            self._memo[op] = (merged[order], np.concatenate([known_values, fresh])[order])
        return values

    # ── backend interface ────────────────────────────────────────────────

    def _store(self, seqs: list) -> np.ndarray:
        """Add host sequences as new tapes; returns their ids."""
        raise NotImplementedError

    def _compute(self, op, x: np.ndarray, y) -> np.ndarray:
        """op on distinct tape ids (x[k], y[k]): new tape ids for run/memory/concat, values otherwise."""
        raise NotImplementedError

    def _sequences(self, ids: np.ndarray) -> list:
        """The content of each tape, as a sequence of ints."""
        raise NotImplementedError


def _distinct_pairs(x: np.ndarray, y: np.ndarray):
    """Sorted distinct keys (x << 32 | y) and each pair's index among them."""
    if not len(x):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    width = int(y.max()) + 1
    span = (int(x.max()) + 1) * width
    if span <= 4 * len(x) + (1 << 22):  # dense: no sort
        flat = x * width + y
        seen = np.zeros(span, dtype=bool)
        seen[flat] = True
        uniq = np.flatnonzero(seen)
        where = np.empty(span, dtype=np.int64)
        where[uniq] = np.arange(len(uniq))
        return ((uniq // width) << 32) | (uniq % width), where[flat]
    return np.unique((x << 32) | y, return_inverse=True)


def _compact(seq):
    """bytes when every symbol fits in a byte, else a tuple; equal contents get the same form."""
    if isinstance(seq, bytes):
        return seq
    try:
        return bytes(seq)
    except (TypeError, ValueError):
        return tuple(seq)


class HostMatchups(Matchups):
    """
    Matchups on any scalar interpreter: tapes are held compactly (bytes when
    every symbol fits in a byte, else tuples) and runs call interpreter.run(),
    or interpreter.run_bytes(code, inp) -> bytes when it has one and both
    tapes are bytes (treemo's C interpreter does, sparing list conversions).
    """

    def __init__(self, interpreter, programs: Sequence, a_ids, b_ids):
        super().__init__(interpreter, a_ids, b_ids)
        self._seqs: list = []
        self._store(programs)

    def _store(self, seqs) -> np.ndarray:
        start = len(self._seqs)
        self._seqs.extend(_compact(s) for s in seqs)
        lengths = np.fromiter((len(s) for s in self._seqs[start:]), dtype=np.int64)
        self._lengths = np.concatenate([self._lengths, lengths])
        return np.arange(start, len(self._seqs), dtype=np.int64)

    def _execute(self, code, inp, result: int):
        fast = getattr(self.interpreter, "run_bytes", None)
        if result == 0 and fast is not None and isinstance(code, bytes) and isinstance(inp, bytes):
            return fast(code, inp)
        return self.interpreter.run(list(code), list(inp))[result]

    def _compute(self, op, x, y) -> np.ndarray:
        s = self._seqs
        if op[0] == "count":
            return np.fromiter((_count(s[i], op[1]) for i in x.tolist()), dtype=np.int64, count=len(x))
        pairs = list(zip(x.tolist(), y.tolist()))
        if op == "run":
            return self._store([self._execute(s[c], s[i], 0) for c, i in pairs])
        if op == "memory":
            return self._store([self._execute(s[c], s[i], 1) for c, i in pairs])
        if op == "concat":
            return self._store([s[p] + s[q] if type(s[p]) is type(s[q]) else tuple(s[p]) + tuple(s[q])
                                for p, q in pairs])
        if op == "lcs":
            return np.fromiter((lcs_length(s[p], s[q]) for p, q in pairs), dtype=np.int64, count=len(pairs))
        if op == "equals":
            return np.fromiter((s[p] == s[q] for p, q in pairs), dtype=np.int64, count=len(pairs))
        raise ValueError(f"unknown operation {op!r}")

    def _sequences(self, ids) -> list:
        return [self._seqs[i] for i in ids.tolist()]


def _count(seq, symbol: int) -> int:
    if isinstance(seq, bytes) and not 0 <= symbol < 256:
        return 0
    return seq.count(symbol)


def play_one(reward: Callable, interpreter, code_a, code_b):
    """reward for the single matchup (code_a, code_b) on a scalar interpreter."""
    programs, a, b = block([code_a], [code_b])
    return np.asarray(reward(HostMatchups(interpreter, programs, a, b)))[0].item()

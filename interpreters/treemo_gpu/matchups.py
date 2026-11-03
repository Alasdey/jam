"""
Matchups on the GPU (treemo): every tape of a batch lives bit-packed in one
device heap, and each operation of core.matchups runs as one batched launch.

Tapes that are not 0/1 sequences (placeholder's staple, say) stay on the
host; any operation involving one, and runs of programs that are not
balanced 0/1 words, are computed on the host instead, with the C interpreter
and core.matchups.lcs_length. Results never depend on which side computed them.
"""

import numpy as np

from core.matchups import Matchups, lcs_length

_THREADS = 128  # 4 warps per block, one job per warp
_LCS_MAX_BITS = 16 * 32 * 32  # the device LCS keeps the shorter side in registers
_THREAD_LCS_WORDS = np.array([4, 8, 16, 24, 32, 48, 64, 96, 128])  # sizes of the one-job-per-thread LCS


def pack(tapes):
    """0/1 tapes as consecutive words, each word-aligned and followed by a zero word: (words, word offsets)."""
    lens = np.fromiter(map(len, tapes), dtype=np.int64, count=len(tapes))
    words = (lens + 31) // 32 + 1
    offsets = np.concatenate(([0], np.cumsum(words)[:-1])).astype(np.int64)
    bits = np.zeros(32 * int(words.sum()), dtype=np.uint8)
    for tape, start in zip(tapes, (32 * offsets).tolist()):
        bits[start:start + len(tape)] = np.frombuffer(tape, dtype=np.uint8)
    return np.packbits(bits, bitorder="little").view("<u4"), offsets


def _exclusive_cumsum(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.int64)
    np.cumsum(values[:-1], out=out[1:])
    return out


class DeviceMatchups(Matchups):
    """Matchups for payoff.backend="cuda"; `stats` counts device and host work."""

    def __init__(self, runtime, programs, a_ids, b_ids):
        super().__init__(runtime, a_ids, b_ids)
        self._rt = runtime
        self._cp = runtime._cuda()
        self._host: list = []  # bytes, or None while a tape is only on the device
        self._dev = np.zeros(0, dtype=np.int64)  # heap word offset, -1 for host-only tapes
        self._heap = self._cp.zeros(1 << 16, dtype=self._cp.uint32)
        self._used = 1  # word 0 is the leading guard
        self.stats = {"gpu_runs": 0, "cpu_runs": 0, "launches": 0, "host_measures": 0}
        self._store(programs)

    # ── tapes ────────────────────────────────────────────────────────────

    def _append(self, lengths, dev, host) -> np.ndarray:
        start = len(self._host)
        self._host.extend(host)
        self._lengths = np.concatenate([self._lengths, np.asarray(lengths, dtype=np.int64)])
        self._dev = np.concatenate([self._dev, np.asarray(dev, dtype=np.int64)])
        return np.arange(start, len(self._host), dtype=np.int64)

    def _reserve(self, words: int) -> int:
        """Heap space for `words` words; may move the heap, so launch only after reserving."""
        need = self._used + words
        if need > len(self._heap):
            grown = self._cp.zeros(max(need, 2 * len(self._heap)), dtype=self._cp.uint32)
            grown[:self._used] = self._heap[:self._used]
            self._heap = grown
        offset, self._used = self._used, need
        return offset

    def _store(self, seqs) -> np.ndarray:
        data = [s if isinstance(s, bytes) else bytes(s) for s in seqs]
        lengths = np.fromiter(map(len, data), dtype=np.int64, count=len(data))
        ids = self._append(lengths, np.full(len(data), -1, dtype=np.int64), data)
        self._upload(ids)
        return ids

    def _upload(self, ids: np.ndarray) -> None:
        """Put the 0/1 tapes among host-only ids on the device."""
        up = [i for i in ids.tolist()
              if self._dev[i] < 0 and not self._host[i].translate(None, b"\x00\x01")]
        if up:
            words, offsets = pack([self._host[i] for i in up])
            base = self._reserve(len(words))
            self._heap[base:base + len(words)] = self._cp.asarray(words)
            self._dev[up] = base + offsets

    def _bytes(self, ids) -> list:
        """Host copies of tapes, downloading those only on the device."""
        ids = np.asarray(ids, dtype=np.int64)
        missing = [i for i in np.unique(ids).tolist() if self._host[i] is None]
        if missing:
            heap = self._heap[:self._used].get() if len(missing) > 64 else None
            for i in missing:
                start, n = int(self._dev[i]), int(self._lengths[i])
                end = start + (n + 31) // 32
                words = heap[start:end] if heap is not None else self._heap[start:end].get()
                self._host[i] = np.unpackbits(np.ascontiguousarray(words).view(np.uint8),
                                              bitorder="little", count=n).tobytes()
        return [self._host[i] for i in ids.tolist()]

    def _sequences(self, ids) -> list:
        return self._bytes(ids)

    # ── operations ───────────────────────────────────────────────────────

    def _compute(self, op, x, y) -> np.ndarray:
        if op == "run":
            return self._run(x, y)
        if op == "memory":
            return x.copy()  # treemo returns the program it ran
        if op == "concat":
            return self._concat(x, y)
        if op == "lcs":
            return self._lcs(x, y)
        if op == "equals":
            return self._equal(x, y)
        if op[0] == "count":
            return self._count(x, op[1])
        raise ValueError(f"unknown operation {op!r}")

    def _launch(self, name: str, n: int, *args) -> None:
        kernel = self._rt._kernels[name]
        kernel(((n * 32 + _THREADS - 1) // _THREADS,), (_THREADS,), args)
        self.stats["launches"] += 1

    def _bits(self, ids: np.ndarray):
        return self._cp.asarray(self._dev[ids] * 32), self._cp.asarray(self._lengths[ids].astype(np.int32))

    def _pairwise_kernel(self, name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = self._cp.empty(len(x), dtype=self._cp.int32)
        self._launch(name, len(x), self._heap, *self._bits(x), *self._bits(y), out, np.int32(len(x)))
        return out.get().astype(np.int64)

    def _on_device(self, *ids: np.ndarray) -> np.ndarray:
        return np.logical_and.reduce([self._dev[i] >= 0 for i in ids])

    def _count(self, x, symbol: int) -> np.ndarray:
        out = np.zeros(len(x), dtype=np.int64)
        dev = self._on_device(x)
        if symbol in (0, 1) and dev.any():  # device tapes hold no other symbol
            ones = np.empty(int(dev.sum()), dtype=np.int64)
            d_out = self._cp.empty(len(ones), dtype=self._cp.int32)
            self._launch("treemo_ones", len(ones), self._heap, *self._bits(x[dev]), d_out, np.int32(len(ones)))
            ones[:] = d_out.get()
            out[dev] = ones if symbol == 1 else self._lengths[x[dev]] - ones
        host = np.flatnonzero(~dev)
        if len(host) and 0 <= symbol < 256:
            out[host] = [t.count(symbol) for t in self._bytes(x[host])]
            self.stats["host_measures"] += len(host)
        return out

    def _equal(self, x, y) -> np.ndarray:
        out = np.zeros(len(x), dtype=np.int64)
        dev = self._on_device(x, y)
        if dev.any():
            out[dev] = self._pairwise_kernel("treemo_equal", x[dev], y[dev])
        host = np.flatnonzero(~dev)
        if len(host):
            out[host] = [p == q for p, q in zip(self._bytes(x[host]), self._bytes(y[host]))]
            self.stats["host_measures"] += len(host)
        return out

    def _lcs(self, x, y) -> np.ndarray:
        out = np.zeros(len(x), dtype=np.int64)
        lx, ly = self._lengths[x], self._lengths[y]
        dev = self._on_device(x, y) & (np.minimum(lx, ly) <= _LCS_MAX_BITS)
        if dev.any():
            swap = lx[dev] < ly[dev]  # the shorter tape is the bit-vector side
            text = np.where(swap, y[dev], x[dev])
            ref = np.where(swap, x[dev], y[dev])
            out[dev] = self._lcs_device(text, ref)
        host = np.flatnonzero(~dev)
        if len(host):
            out[host] = [lcs_length(p, q) for p, q in zip(self._bytes(x[host]), self._bytes(y[host]))]
            self.stats["host_measures"] += len(host)
        return out

    def _lcs_device(self, text, ref) -> np.ndarray:
        """Refs up to 2,048 bits one job per thread, grouped by size in words; longer ones one per warp."""
        out = np.empty(len(text), dtype=np.int64)
        words = (self._lengths[ref] + 31) // 32
        bucket = np.searchsorted(_THREAD_LCS_WORDS, words)  # == len(_THREAD_LCS_WORDS) past 32 words
        for b, k in enumerate(_THREAD_LCS_WORDS):
            jobs = np.flatnonzero(bucket == b)
            if len(jobs):
                d_out = self._cp.empty(len(jobs), dtype=self._cp.int32)
                self._rt._kernels["treemo_lcs_small"](
                    ((len(jobs) + _THREADS - 1) // _THREADS,), (_THREADS,),
                    (self._heap, *self._bits(text[jobs]), *self._bits(ref[jobs]), d_out,
                     np.int32(len(jobs)), np.int32(k)))
                self.stats["launches"] += 1
                out[jobs] = d_out.get()
        jobs = np.flatnonzero(bucket == len(_THREAD_LCS_WORDS))
        if len(jobs):
            out[jobs] = self._pairwise_kernel("treemo_lcs", text[jobs], ref[jobs])
        return out

    def _concat(self, x, y) -> np.ndarray:
        n = self._lengths[x] + self._lengths[y]
        dev = self._on_device(x, y)
        offsets = np.full(len(x), -1, dtype=np.int64)
        host = [None] * len(x)
        if dev.any():
            words = (n[dev] + 31) // 32 + 1
            offsets[dev] = self._reserve(int(words.sum())) + _exclusive_cumsum(words)
            self._launch("treemo_concat", int(dev.sum()), self._heap, *self._bits(x[dev]), *self._bits(y[dev]),
                         self._cp.asarray(offsets[dev]), np.int32(dev.sum()))
        rest = np.flatnonzero(~dev)
        for k, p, q in zip(rest.tolist(), self._bytes(x[rest]), self._bytes(y[rest])):
            host[k] = p + q
        ids = self._append(n, offsets, host)
        self._upload(ids[rest])
        return ids

    def _run(self, codes, inputs) -> np.ndarray:
        rt = self._rt
        n = len(codes)
        distinct = np.unique(codes)
        tables = rt._rule_tables(self._bytes(distinct))
        empty = (np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.uint8))
        valid = np.array([t is not None for t in tables], dtype=bool)
        tables = [t if t is not None else empty for t in tables]
        nrules = np.array([len(t[0]) for t in tables], dtype=np.int64)
        growth = np.array([max(0, int((t[0][:, 3] - t[0][:, 1]).max())) if len(t[0]) else 0
                           for t in tables], dtype=np.int64)
        which = np.searchsorted(distinct, codes)

        # Exact state bound: no firing grows the tape by more than its program's
        # largest rule growth (C runs no step at all when max_step <= 0).
        cap = self._lengths[inputs] + max(rt.max_step, 0) * growth[which]
        state_words = (cap + 31) // 32 + 2
        per_job = 4 * (state_words + nrules[which]) + 64
        budget = rt.memory_mb * 2**20
        dev = valid[which] & self._on_device(inputs) & (cap < 2**31) & (per_job <= budget)

        out_len = np.zeros(n, dtype=np.int64)
        out_dev = np.full(n, -1, dtype=np.int64)
        out_host = [None] * n
        jobs = np.flatnonzero(dev)
        if len(jobs):
            cp = self._cp
            d_rules = cp.asarray(np.ascontiguousarray(np.concatenate([t[0] for t in tables])))
            d_ident = cp.asarray(np.concatenate([t[1] for t in tables]))
            rule_start = _exclusive_cumsum(nrules)
            # Launches bounded by the memory budget; jobs keep their order, which
            # is program-major for blocks and keeps a program's rules in cache.
            used = np.cumsum(per_job[jobs])
            first = 0
            while first < len(jobs):
                limit = (used[first - 1] if first else 0) + max(budget, per_job[jobs[first]])
                last = max(first + 1, int(np.searchsorted(used, limit, side="right")))
                chunk = jobs[first:last]
                out_len[chunk], out_dev[chunk] = self._run_launch(
                    codes[chunk], inputs[chunk], which[chunk], rule_start, nrules,
                    cap[chunk], state_words[chunk], d_rules, d_ident)
                first = last
        rest = np.flatnonzero(~dev)
        for k, code, inp in zip(rest.tolist(), self._bytes(codes[rest]), self._bytes(inputs[rest])):
            out_host[k] = rt._c.run_bytes(code, inp)
            out_len[k] = len(out_host[k])
        self.stats["gpu_runs"] += len(jobs)
        self.stats["cpu_runs"] += len(rest)
        ids = self._append(out_len, out_dev, out_host)
        self._upload(ids[rest])
        return ids

    def _run_launch(self, codes, inputs, which, rule_start, nrules, cap, state_words, d_rules, d_ident):
        cp, rt = self._cp, self._rt
        n = len(codes)
        state_off = _exclusive_cumsum(state_words)
        dirty = nrules[which]
        d_state = cp.asarray(state_off)
        d_pool = cp.empty(int(state_words.sum()), dtype=cp.uint32)
        d_dpool = cp.empty(max(1, int(dirty.sum())), dtype=cp.int32)
        d_len = cp.empty(n, dtype=cp.int32)
        self._launch(
            "treemo_run", n, self._heap, d_rules, d_ident,
            cp.asarray(self._dev[codes] * 32), cp.asarray(rule_start[which].astype(np.int32)),
            cp.asarray(dirty.astype(np.int32)),
            cp.asarray(self._dev[inputs] * 32), cp.asarray(self._lengths[inputs].astype(np.int32)),
            d_state, cp.asarray(cap.astype(np.int32)), cp.asarray(_exclusive_cumsum(dirty)),
            d_pool, d_dpool, d_len,
            np.int32(n), np.int32(rt.max_step), np.int32(rt.pass_mode), np.int32(rt.first_mode),
        )
        lengths = d_len.get().astype(np.int64)
        if (lengths < 0).any():
            raise RuntimeError("Treemo CUDA state overflowed its exact growth bound")
        words = (lengths + 31) // 32 + 1
        dst = self._reserve(int(words.sum())) + _exclusive_cumsum(words)
        self._launch("treemo_gather", n, d_state, d_len, cp.asarray(dst), d_pool, self._heap, np.int32(n))
        return lengths, dst

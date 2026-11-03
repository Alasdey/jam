"""
Treemo on CUDA: one warp per (program, input) execution.

Results are identical to interpreters/treemo_c (the reference in tests): the
kernel replays the C interpreter's rule advancement and scan watermarks.
Programs and inputs are tapes of a TapeStore (core/matchups.py); rewards and
creation reach this through core.matchups.Matchups.run.
"""

import numpy as np

from core.device import cupy, exclusive_cumsum, kernel, warp_launch

_SOURCE = "interpreters/treemo_gpu/kernels.cu"


class TreemoRuntime:
    """
    Treemo settings plus the execution of batches of jobs. memory_mb bounds
    the device memory of one launch's states and scan watermarks; bigger
    batches are split into several launches.
    """

    def __init__(self, max_step: int, pass_mode: bool, first_mode: bool, memory_mb: int = 1024):
        if memory_mb < 1:
            raise ValueError("memory_mb must be positive")
        self.max_step = int(max_step)
        self.pass_mode = bool(pass_mode)
        self.first_mode = bool(first_mode)
        self.memory_mb = int(memory_mb)

    @classmethod
    def from_config(cls, exp_cfg) -> "TreemoRuntime":
        t = exp_cfg.treemo
        return cls(t.max_step, t.pass_mode, t.first_mode, exp_cfg.payoff.gpu_memory_mb)

    def execute(self, store, rules, codes, inputs):
        """
        Run program codes[k] on input inputs[k] (tape ids of `store`); rules
        holds each code's rule rows (rules.start/count/growth indexed by tape
        id). Returns the outputs' new tape ids.
        """
        cp = cupy()
        n = len(codes)
        if n == 0:
            return cp.zeros(0, dtype=cp.int64)
        nrules = rules.count[codes]
        # Exact state bound: no firing grows the tape by more than its program's
        # largest rule growth (C runs no step at all when max_step <= 0).
        cap = store.length[inputs] + max(self.max_step, 0) * rules.growth[codes]
        if bool((cap >= 2**31).any()):
            raise RuntimeError("a Treemo state could exceed 2**31 bits")
        state_words = (cap + 31) // 32 + 2
        per_job = 4 * (state_words + nrules) + 64
        budget = self.memory_mb * 2**20

        # Launches of about `budget` bytes, jobs kept in order.
        used = cp.cumsum(per_job)
        total = int(used[-1])
        cuts = [0]
        if total > budget:
            marks = cp.arange(budget, total, budget, dtype=cp.int64)
            cuts += sorted(set(cp.asnumpy(cp.searchsorted(used, marks, side="right")).tolist()) - {0, n})
        cuts.append(n)
        out = cp.empty(n, dtype=cp.int64)
        for a, b in zip(cuts[:-1], cuts[1:]):
            out[a:b] = self._launch(store, rules, codes[a:b], inputs[a:b], nrules[a:b], cap[a:b],
                                    state_words[a:b])
        return out

    def _launch(self, store, rules, codes, inputs, nrules, cap, state_words):
        cp = cupy()
        n = len(codes)
        state_off = exclusive_cumsum(state_words)
        dirty_off = exclusive_cumsum(nrules)
        pool = cp.empty(int(state_words.sum()), dtype=cp.uint32)
        dpool = cp.empty(max(1, int(nrules.sum())), dtype=cp.int32)
        out_len = cp.empty(n, dtype=cp.int32)
        warp_launch(
            kernel(_SOURCE, "treemo_run"), n,
            store.heap, rules.table, rules.ident,
            32 * store.offset[codes], rules.start[codes].astype(cp.int32), nrules.astype(cp.int32),
            32 * store.offset[inputs], store.length[inputs].astype(cp.int32),
            state_off, cap.astype(cp.int32), dirty_off, pool, dpool, out_len,
            np.int32(n), np.int32(self.max_step), np.int32(self.pass_mode), np.int32(self.first_mode),
        )
        if bool((out_len < 0).any()):
            raise RuntimeError("Treemo CUDA state overflowed its exact growth bound")
        lengths = out_len.astype(cp.int64)
        words = (lengths + 31) // 32 + 1
        ids, dst = store.allocate(lengths, words)
        warp_launch(kernel(_SOURCE, "treemo_gather"), n, state_off, out_len, dst, pool, store.heap, np.int32(n))
        return ids

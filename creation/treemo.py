"""
Creation of Treemo programs on the GPU: random trees and offspring.

One generation's newcomers are made in one pass (TreemoCreator.create):
fresh random trees, then offspring of the survivors, each made like
creation/genetics.py used to on the CPU:

    with crossover_prob (and at least two survivors): two distinct parents;
        with homoiconic_prob, the child is parent A's output run on B
        (homoiconic "output"), unless that output is empty;
        otherwise the configured tree crossover of A and B;
    otherwise: the configured tree mutation of one survivor.

The operators run in creation/creation.cu, one thread per child; the
homoiconic runs go through core.matchups like any other execution.
"""

import numpy as np

from config import ExperimentConfig
from core.device import cupy, exclusive_cumsum, kernel, launch, warp_launch
from core.matchups import Matchups
from core.population import METHOD_CODE, Population

_SOURCE = "creation/creation.cu"
_POPULATION = "core/cuda/population.cu"

OP_RANDOM = 0
# Room for a subtree-mutation child: factor x parent length + extra symbols.
# Children that outgrow it are made again with 4x the room, same draws.
_MUTATION_ROOM = (3, 256)
TREE_MUTATION_OPS = {"subtree": 1, "leaf": 2}
TREE_CROSSOVER_OPS = {"random_depth": 3, "depth1": 4}


class TreemoCreator:
    def __init__(self, cfg: ExperimentConfig, runtime):
        gc = cfg.genetics
        if not 0 <= gc.mutation_rate <= 1:
            raise ValueError("mutation rate must be between 0 and 1")
        self.tree_size = cfg.treemo.tree_size
        self.rate = gc.mutation_rate
        self.crossover_prob = gc.crossover_prob
        self.homoiconic_prob = gc.homoiconic_prob
        self.mutate_op = TREE_MUTATION_OPS[gc.tree_mutation_op]
        self.cross_op = TREE_CROSSOVER_OPS[gc.tree_crossover_op]
        self.runtime = runtime

    def create(self, survivors: Population, n_random: int, n_offspring: int, rng, gen: int,
               first_id: int) -> Population:
        """n_random random trees, then n_offspring children of survivors (random trees when there are none)."""
        n = n_random + n_offspring
        ops = np.full(n, OP_RANDOM, dtype=np.int32)
        a = np.full(n, -1, dtype=np.int64)
        b = np.full(n, -1, dtype=np.int64)
        method = np.full(n, METHOD_CODE["random"], dtype=np.int8)
        homo = np.zeros(n, dtype=bool)
        n_surv = len(survivors)
        if n_offspring and n_surv:
            cp = cupy()
            cross = cp.asnumpy(rng.uniform(n_offspring) < self.crossover_prob) & (n_surv >= 2)
            pa = cp.asnumpy(rng.integers(n_surv, n_offspring))
            shift = cp.asnumpy(rng.integers(max(n_surv - 1, 1), n_offspring))
            pb = (pa + 1 + shift) % n_surv  # a distinct second parent, uniformly
            try_homo = cross & cp.asnumpy(rng.uniform(n_offspring) < self.homoiconic_prob)
            pm = cp.asnumpy(rng.integers(n_surv, n_offspring))
            off = slice(n_random, n)
            ops[off] = np.where(cross, self.cross_op, self.mutate_op)
            a[off] = np.where(cross, pa, pm)
            b[off] = np.where(cross, pb, -1)
            method[off] = np.where(cross, METHOD_CODE["crossover"], METHOD_CODE["mutate"])
            homo[off] = try_homo

        outputs = None
        if homo.any():
            m = Matchups(self.runtime, survivors, a[homo], b[homo])
            out = m.run(m.a, m.b)
            nonempty = cupy().asnumpy(out.length() > 0)
            idx = np.flatnonzero(homo)
            homo[idx[~nonempty]] = False  # empty output: plain crossover instead
            method[idx[nonempty]] = METHOD_CODE["homoiconic"]
            outputs = (m._store, cupy().asnumpy(out.ids)[nonempty])

        lengths, heap_parts = self._run_operators(survivors, ops, a, b, ~homo, rng)
        return self._assemble(survivors, n, lengths, heap_parts, homo, outputs, method, a, b, gen, first_id)

    def _run_operators(self, survivors, ops, a, b, jobs_mask, rng):
        """Lengths of every kernel-made child, and (scratch, at, job index) to pack them from."""
        cp = cupy()
        jobs = np.flatnonzero(jobs_mask)
        lengths = np.zeros(len(ops), dtype=np.int64)
        if not len(jobs):
            return lengths, None
        surv_len = survivors.lengths() if len(survivors) else np.zeros(0, dtype=np.int64)
        la = np.where(a[jobs] >= 0, surv_len[np.maximum(a[jobs], 0)] if len(surv_len) else 0, 0)
        lb = np.where(b[jobs] >= 0, surv_len[np.maximum(b[jobs], 0)] if len(surv_len) else 0, 0)
        op = ops[jobs]
        cap = np.select(
            [op == OP_RANDOM, op == 1, op == 2],
            [np.full(len(jobs), max(2 * (self.tree_size - 1), 0)), _MUTATION_ROOM[0] * la + _MUTATION_ROOM[1],
             la + 2],
            la + lb,
        ).astype(np.int64)
        extra = np.where(op >= 3, la + lb, 0)
        args = rng.kernel_args()  # one call: retries replay the same draws
        heap = survivors.heap if len(survivors) else cp.zeros(1, dtype=cp.uint32)
        surv_off = survivors.offset if len(survivors) else cp.zeros(1, dtype=cp.int64)
        surv_length = survivors.length if len(survivors) else cp.zeros(1, dtype=cp.int32)
        todo = np.arange(len(jobs))
        scratch_parts = []
        while len(todo):
            if (cap[todo] + extra[todo] >= 2**31).any():
                raise RuntimeError("a child would exceed 2**31 symbols")
            at = exclusive_cumsum(cp.asarray(cap[todo] + extra[todo]))
            scratch = cp.empty(max(1, int((cap[todo] + extra[todo]).sum())), dtype=cp.uint8)
            out_len = cp.empty(len(todo), dtype=cp.int32)
            launch(kernel(_SOURCE, "create"), len(todo),
                   heap, surv_off, surv_length, np.int32(len(todo)),
                   cp.asarray(op[todo]), cp.asarray(a[jobs[todo]]), cp.asarray(b[jobs[todo]]),
                   cp.asarray(jobs[todo].astype(np.uint32)),
                   scratch, at, cp.asarray(cap[todo].astype(np.int32)), out_len,
                   np.int32(self.tree_size), np.float64(self.rate), *args)
            got = cp.asnumpy(out_len).astype(np.int64)
            done = got >= 0
            lengths[jobs[todo[done]]] = got[done]
            scratch_parts.append((scratch, cp.asnumpy(at)[done], jobs[todo[done]]))
            cap[todo[~done]] *= 4
            todo = todo[~done]
        return lengths, scratch_parts

    def _assemble(self, survivors, n, lengths, scratch_parts, homo, outputs, method, a, b, gen, first_id):
        cp = cupy()
        if outputs is not None:
            store, out_ids = outputs
            lengths[np.flatnonzero(homo)] = cp.asnumpy(store.length[cp.asarray(out_ids)])
        words = (lengths + 31) // 32 + 1
        dst = 1 + np.concatenate(([0], np.cumsum(words)[:-1])).astype(np.int64)
        heap = cp.zeros(1 + int(words.sum()), dtype=cp.uint32)
        for scratch, at, which in scratch_parts or ():
            warp_launch(kernel(_SOURCE, "pack_children"), len(which), scratch, cp.asarray(at),
                        cp.asarray(lengths[which].astype(np.int32)), heap, cp.asarray(dst[which]),
                        np.int32(len(which)))
        if outputs is not None:
            which = np.flatnonzero(homo)
            src = cp.asarray(out_ids)
            warp_launch(kernel(_POPULATION, "copy_tapes"), len(which), store.heap, store.offset[src],
                        store.length[src].astype(cp.int32), heap, cp.asarray(dst[which]), np.int32(len(which)))
        parents = np.full((n, 2), -1, dtype=np.int64)
        if len(survivors):
            parents[:, 0] = np.where(a >= 0, survivors.ids[np.maximum(a, 0)], -1)
            parents[:, 1] = np.where(b >= 0, survivors.ids[np.maximum(b, 0)], -1)
        return Population(heap, cp.asarray(dst), cp.asarray(lengths.astype(np.int32)),
                          np.arange(first_id, first_id + n, dtype=np.int64), method, parents,
                          np.full(n, gen, dtype=np.int32))

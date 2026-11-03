"""
A population of Treemo programs on the device, with its records on the host.

Genomes live packed in one heap of uint32 words (core/cuda/tapes.cuh: word
aligned, zero padded, a zero word after each, a zero word first), indexed by
`offset` (word offset) and `length` (bits). That is the layout the Treemo
kernels read, so genomes never leave the device during a generation; they
come back only to be logged or published (`genomes()`).

Records (id, method, parents, birth generation) are small and only needed for
logging, so they stay on the host, aligned with the genomes.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.device import cupy, exclusive_cumsum, kernel, launch, warp_launch

_SOURCE = "core/cuda/population.cu"

METHODS = ("random", "mutate", "crossover", "homoiconic", "seeded")
METHOD_CODE = {name: code for code, name in enumerate(METHODS)}


def pack(genomes: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """0/1 sequences as a heap: (words with the leading zero word, word offset of each)."""
    lens = np.fromiter(map(len, genomes), dtype=np.int64, count=len(genomes))
    words = (lens + 31) // 32 + 1
    offsets = 1 + np.concatenate(([0], np.cumsum(words)[:-1])).astype(np.int64)
    bits = np.zeros(32 * (1 + int(words.sum())), dtype=np.uint8)
    for genome, start in zip(genomes, (32 * offsets).tolist()):
        seq = np.asarray(genome, dtype=np.uint8)
        if seq.size and seq.max() > 1:
            raise ValueError("Treemo tapes hold only the symbols 0 and 1")
        bits[start:start + len(seq)] = seq
    return np.packbits(bits, bitorder="little").view("<u4").copy(), offsets


def unpack(words: np.ndarray, offsets: np.ndarray, lengths: np.ndarray) -> list[list[int]]:
    """The inverse of pack, from a host copy of a heap."""
    as_bytes = np.ascontiguousarray(words).view(np.uint8)
    out = []
    for off, n in zip(offsets.tolist(), lengths.tolist()):
        out.append(np.unpackbits(as_bytes[4 * off:4 * off + 4 * ((n + 31) // 32)],
                                 bitorder="little", count=n).tolist())
    return out


@dataclass
class Population:
    heap: object         # cupy uint32
    offset: object       # cupy int64, word offset of each genome
    length: object       # cupy int32, bits
    ids: np.ndarray      # int64
    method: np.ndarray   # int8, index into METHODS
    parents: np.ndarray  # int64 (n, 2), -1 where there is no parent
    born_gen: np.ndarray  # int32

    def __len__(self) -> int:
        return len(self.ids)

    # ── construction ─────────────────────────────────────────────────────

    @classmethod
    def empty(cls) -> "Population":
        cp = cupy()
        return cls(cp.zeros(1, dtype=cp.uint32), cp.zeros(0, dtype=cp.int64), cp.zeros(0, dtype=cp.int32),
                   np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8),
                   np.full((0, 2), -1, dtype=np.int64), np.zeros(0, dtype=np.int32))

    @classmethod
    def from_genomes(cls, genomes: Sequence[Sequence[int]], ids, method, parents, born_gen) -> "Population":
        """Upload host genomes; `method` may be names or codes, `parents` lists of 0-2 ids."""
        cp = cupy()
        words, offsets = pack(genomes)
        n = len(genomes)
        return cls(
            cp.asarray(words), cp.asarray(offsets),
            cp.asarray(np.fromiter(map(len, genomes), dtype=np.int32, count=n)),
            np.asarray(ids, dtype=np.int64).reshape(n),
            _method_codes(method, n),
            _parent_table(parents, n),
            np.broadcast_to(np.asarray(born_gen, dtype=np.int32), (n,)).copy(),
        )

    @classmethod
    def from_records(cls, records: Sequence[dict]) -> "Population":
        """From births-file style records: {"id", "genome", "method", "parents", "born_gen"}."""
        return cls.from_genomes([r["genome"] for r in records], [r["id"] for r in records],
                                [r["method"] for r in records], [r["parents"] for r in records],
                                [r["born_gen"] for r in records])

    def take(self, idx) -> "Population":
        """The individuals at idx (host or device indices), in that order, in a compact heap."""
        cp = cupy()
        idx_host = np.asarray(cp.asnumpy(idx) if isinstance(idx, cp.ndarray) else idx, dtype=np.int64)
        idx_dev = cp.asarray(idx_host)
        length = self.length[idx_dev]
        words = (length.astype(cp.int64) + 31) // 32 + 1
        dst = 1 + exclusive_cumsum(words)
        total = 1 + int(words.sum()) if len(idx_host) else 1
        heap = cp.zeros(total, dtype=cp.uint32)
        warp_launch(kernel(_SOURCE, "copy_tapes"), len(idx_host),
                    self.heap, self.offset[idx_dev], length, heap, dst, np.int32(len(idx_host)))
        return Population(heap, dst, length, self.ids[idx_host], self.method[idx_host],
                          self.parents[idx_host], self.born_gen[idx_host])

    @staticmethod
    def concat(parts: Sequence["Population"]) -> "Population":
        """One population holding every part's individuals, in order."""
        cp = cupy()
        parts = [p for p in parts if len(p)]
        if not parts:
            return Population.empty()
        if len(parts) == 1:
            return parts[0]
        heap = cp.concatenate([parts[0].heap] + [p.heap[1:] for p in parts[1:]])
        shifts = np.cumsum([0] + [len(p.heap) - 1 for p in parts[:-1]])
        offset = cp.concatenate([p.offset + int(s) for p, s in zip(parts, shifts)])
        return Population(
            heap, offset, cp.concatenate([p.length for p in parts]),
            np.concatenate([p.ids for p in parts]), np.concatenate([p.method for p in parts]),
            np.concatenate([p.parents for p in parts]), np.concatenate([p.born_gen for p in parts]),
        )

    # ── reading ──────────────────────────────────────────────────────────

    def lengths(self) -> np.ndarray:
        return cupy().asnumpy(self.length).astype(np.int64)

    def genomes(self, idx=None) -> list[list[int]]:
        """Host copies of the genomes (all, or those at idx), as lists of ints."""
        cp = cupy()
        sub = self if idx is None else self.take(idx)
        return unpack(cp.asnumpy(sub.heap), cp.asnumpy(sub.offset), sub.lengths())

    def hashes(self):
        """64-bit hash of each genome (device uint64); equal genomes hash equal."""
        cp = cupy()
        out = cp.empty(len(self), dtype=cp.uint64)
        launch(kernel(_SOURCE, "hash_tapes"), len(self), self.heap, self.offset, self.length, out, np.int32(len(self)))
        return out

    def equal(self, x, y):
        """Device bool array: genome x[k] equals genome y[k]."""
        cp = cupy()
        x = cp.asarray(x, dtype=cp.int64)
        y = cp.asarray(y, dtype=cp.int64)
        out = cp.empty(len(x), dtype=cp.uint8)
        warp_launch(kernel(_SOURCE, "equal_tapes"), len(x), self.heap, self.offset, self.length, x, y, out,
                    np.int32(len(x)))
        return out.astype(bool)

    def records(self, idx=None) -> list[dict]:
        """Births-file records (id, genome, method, parents, born_gen) of all or idx."""
        idx = np.arange(len(self)) if idx is None else np.asarray(idx, dtype=np.int64)
        genomes = self.genomes(idx)
        return [
            {"id": int(self.ids[i]), "genome": g, "method": METHODS[self.method[i]],
             "parents": [int(p) for p in self.parents[i] if p >= 0], "born_gen": int(self.born_gen[i])}
            for i, g in zip(idx.tolist(), genomes)
        ]


def _method_codes(method, n: int) -> np.ndarray:
    if isinstance(method, str):
        return np.full(n, METHOD_CODE[method], dtype=np.int8)
    values = list(method)
    if values and isinstance(values[0], str):
        return np.array([METHOD_CODE[m] for m in values], dtype=np.int8)
    return np.asarray(values, dtype=np.int8).reshape(n)


def _parent_table(parents, n: int) -> np.ndarray:
    if isinstance(parents, np.ndarray) and parents.ndim == 2:
        return parents.astype(np.int64)
    table = np.full((n, 2), -1, dtype=np.int64)
    for i, ps in enumerate(parents):
        if len(ps) > 2:
            raise ValueError("an individual has at most two parents")
        table[i, :len(ps)] = ps
    return table

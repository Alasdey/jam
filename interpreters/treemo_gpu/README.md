# treemo_gpu

Treemo on CUDA, the only way jam runs programs. Results are identical to
`interpreters/treemo_c` (same rules, same leftmost matches, same scan order),
checked bit for bit in every mode by `tests/test_gpu_engine.py`.

| File | Role |
|---|---|
| `kernels.cu` | execution (`treemo_run`, one warp per job), output gathering, equality, counts, concatenation, LCS |
| `rules.py`, `rules.cu` | rule extraction for many programs at once, on the device |
| `treemo.py` | `TreemoRuntime`: settings plus the execution of batches of jobs within a memory budget |

Rewards and creation reach it through `core.matchups.Matchups`; nothing else
calls it directly.

## How it works

- **One heap per batch.** Every tape (programs, constants, outputs) lives
  bit-packed, 32 symbols per word, in one device array
  ([core/cuda/tapes.cuh](../../core/cuda/tapes.cuh)); kernels address tapes by
  offset and length, so programs, inputs and outputs are interchangeable.
- **One warp per execution.** A pattern search tests 1,024 start positions per
  step (32 per lane, all candidates of a lane dropped together bit by bit); a
  ballot picks the leftmost match. A rewrite shifts the tail with funnel shifts
  across the warp, in place.
- **Same scan watermarks as C** (`dirty[]`): a rule resumes scanning where it
  last failed or fired. On evolved genomes this cuts scanning 40–60×.
- **Exact state bounds.** Each execution gets `len(input) + max_step × (largest
  growth of one rule)` bits, which no execution can exceed, so nothing is
  truncated and nothing is retried.
- **Rules** are extracted with vectorized CuPy over all the programs' bits
  (segmented depths, open/close matching by a sort on (program, level),
  children counted one level down), in chunks of 2²⁴ bits. Unbalanced
  programs are refused: the C interpreter's behaviour on them is undefined.
- **LCS on the device**: bit-parallel over the shorter tape (Crochemore et al.
  2001), one job per thread up to 4,096 bits, one per warp up to 16,384 bits.
  Longer pairs (both tapes) run in Python on the host, with a loud banner.

## Performance

RTX 5060 Laptop GPU, `max_step=50`, real populations from `outputs/main`
(2026-11-03). One generation = 100 random + 900 offspring, 200 survivors.

| Survivors' median genome | Creation, CPU → GPU | Payoff (blind), GPU |
|---|---|---|
| 807 bits | 1–5 s → 0.05–0.13 s | 0.2 s |
| 5,143 bits | 23–32 s → 0.25–0.76 s | 1.5–1.7 s |
| 8,550 bits | 49–57 s → 0.74–1.52 s | 1.5–1.6 s |

Whole generations of the loop, seeded from stored populations: 30 generations
in 40 s from the lexicase population (8,550-bit genomes; the CPU run took
180 s per generation) and in 17 s from the best_worst_random one. Payoff time
is mostly the Treemo kernel itself; `quine_pressure` adds the LCS kernel
(about 3–4 s per generation on 8,550-bit genomes).

The first generations of a fresh environment take a few extra seconds while
CuPy compiles kernels; it caches them on disk.

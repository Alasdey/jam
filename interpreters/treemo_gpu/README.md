# treemo_gpu

Batched CUDA execution of Treemo, used by `payoff.backend = "cuda"`. Results
are identical to `treemo_c`: same rules, same leftmost matches, same scan
order, checked bit for bit by `tests/test_treemo_gpu.py` and, through every
reward and every batch operation, by `tests/test_payoff_gpu.py`.

## Setup

Needs CuPy built for your CUDA driver (`cupy-cuda13x` or `cupy-cuda12x`;
`nvidia-smi` shows the driver's CUDA version) and the C library, which runs
single executions and anything the device does not take:

```bash
gcc -O3 -shared -fPIC -o interpreters/treemo_c/libtreemo.so interpreters/treemo_c/treemo.c
```

CuPy is imported on first device use only; CPU runs never load it. The
kernels compile on first use (about a second) and are cached by CuPy.

## Use

```python
cfg.experiment.payoff.backend = "cuda"     # treemo interpreter only
cfg.experiment.payoff.gpu_memory_mb = 1024  # per execution launch; bigger batches split
```

Rewards need nothing GPU-specific: they are written against the batch API of
`core/matchups.py` (see "Writing a reward" in the main README), and
`PayoffEngine.matrix()` hands each payoff block to them as one
`DeviceMatchups` (`matchups.py` here) instead of the CPU's `HostMatchups`;
`n_workers` and `chunksize` are unused.

| Operation | On the device |
|---|---|
| `run` | the execution kernel, each distinct (program, input) pair once per batch |
| `equals`, `count`, `lcs`, `similarity`, `concat` | one batched kernel each |
| `length`, `memory` | free: lengths are known, treemo's second result is the program |
| `const`, `map` | host; `map` downloads the tapes it needs, so it is the slow path |

Tapes stay on the device between operations, so `run(run(m.a, m.b), m.a)` or
`m.run(m.a, m.b).lcs(m.b)` never copy outputs back. What the device does not
take is computed on the host, with identical results: runs whose input is not
a 0/1 sequence (`placeholder`'s staple) or whose program is not a balanced 0/1
word go to the C interpreter, and comparisons involving such tapes, or LCS of
two tapes both longer than 16,384 bits, use `core.matchups`. `stats` on a
`DeviceMatchups` counts device runs, C runs, launches and host measurements.

Direct use, for pairs rather than blocks:

```python
from interpreters.treemo_gpu.treemo import TreemoInterpreter

gpu = TreemoInterpreter(max_step=50, pass_mode=False, first_mode=False)
gpu.run_batch(codes, inputs)          # [(output, code), ...] like run()
gpu.lengths(codes, inputs)            # output lengths
gpu.self_similarities(codes, inputs)  # LCS(output, code) / len(code)
gpu.matchups(programs, a_ids, b_ids)  # a DeviceMatchups, as core.matchups.block() returns
```

## How it works

- **One heap per batch.** Every tape (programs, constants, outputs) lives
  bit-packed, 32 symbols per word, in one device array; kernels address tapes
  by offset and length, so programs, inputs and outputs are interchangeable.
- **One warp per execution.** A pattern search tests 1,024 start positions per
  step (32 per lane, all candidates of a lane dropped together bit by bit); a
  ballot picks the leftmost match. A rewrite shifts the tail with funnel shifts
  across the warp, in place.
- **Same scan watermarks as C** (`dirty[]`): a rule resumes scanning where it
  last failed or fired. On evolved genomes this cuts scanning 40–60×, and
  most scans fire within a few dozen positions.
- **Exact state bounds.** Each execution gets `len(input) + max_step × (largest
  growth of one rule)` bits, which no execution can exceed, so nothing is
  truncated and nothing is retried.
- **LCS on the device**: bit-parallel over the shorter tape (Crochemore et al.
  2001), its words spread over the warp, with inter-lane carries resolved by
  one carry-lookahead over two ballots per symbol; it stops as soon as the
  whole shorter tape is matched.
- **Rules** are extracted for all new programs at once with numpy (a rule is
  a two-child node with no two-child ancestor, in pre-order) and cached per
  genome. Patterns and replacements are read straight out of the packed
  program; nothing is copied. Outputs used as programs are downloaded for
  extraction.

## Performance

RTX 5060 Laptop GPU vs Ryzen AI 9 HX 370 (2026-10-05), `max_step=50`, exhaust
mode, genomes from `outputs/main` runs (median 374–1,018 bits, up to 9,900).
CPU-side numbers moved up to 4× with the laptop's power state during the
session; kernel times do not depend on it.

| 1,000 genomes, all 10⁶ ordered executions | C, one core | CUDA end to end | CUDA kernels |
|---|---|---|---|
| output lengths | 18–70 µs per execution: 18–70 s | 0.10–0.17 s | 0.05–0.06 s |
| plus LCS self-similarity (`quine_pressure`) | add 0.1–0.5 s per similarity (pure-Python LCS) | 0.31–0.41 s | 0.18–0.30 s |

Whole generations of the `main` preset (100 random + 900 offspring, KNN-score
to 200), payoff time per generation, results identical on both backends:

| Reward | CPU, 15 workers | CUDA |
|---|---|---|
| `blind` | 8.7–11.6 s early on; ~48 s once genomes grow (logged runs) | 0.01–0.04 s |
| `quine_pressure` | 158–290 s for a 100-individual toy config; full size is out of reach | 0.28–0.83 s |

The LCS stops early once the whole program is matched: about 1% of jobs grow
outputs of 100k–485k bits, and in the worst generation measured the early exit
cut LCS time from 3.3 s to 0.13 s (execution itself took 0.07 s).

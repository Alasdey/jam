# jam

A jam trying to reach escape velocity.

Programs are Treemo trees: balanced words of 0/1 symbols whose two-child
nodes are rewrite rules. Pairs of programs are run against each other,
scored by a reward function into a payoff matrix, and the population evolves
over generations via random injection, mutation, crossover, and
"homoiconic" recombination (using a program's own runtime output as genetic
material). Selection thins the population with a pipeline of steps (lexicase,
best/worst/random, kNN novelty, iterated elimination of dominated
strategies, ...).

Everything runs on one NVIDIA GPU through CuPy: creation, execution, rewards,
selection. The population and its payoff matrix stay on the device for the
whole run; only what is logged comes back to the host.

## Setup

### 1. Python dependencies
```bash
uv sync
```
CuPy must be built for your CUDA driver (`cupy-cuda13x` or `cupy-cuda12x`;
`nvidia-smi` shows the driver's CUDA version). Kernels compile on first use
(a few seconds per process the first time; CuPy caches them on disk).

### 2. Treemo C library (test reference only)
```bash
gcc -O3 -shared -fPIC -o interpreters/treemo_c/libtreemo.so interpreters/treemo_c/treemo.c
```
The tests check the GPU kernels against it; runs do not use it.

### 3. Run
```bash
uv run python main.py                  # "main" preset
uv run python main.py evolution        # any preset from config.PRESETS
uv run pytest                          # test suite
```

## Project layout

| Path | Role |
|---|---|
| `config.py` | All dataclass configs and run presets (`PRESETS`) |
| `main.py` | Thin dispatcher: `python main.py [preset]` → `core.loop.run` |
| `core/` | `loop.py` (the generation loop, resume, checkpointing), `population.py` (genomes on the device, records on the host), `matchups.py` (the batch API rewards are written against), `rng.py` (Philox random numbers), `device.py` (CuPy, kernel loading, loud warnings), `cuda/` (shared CUDA headers), `config_keys.py` (`compat_key`, `method_key`, `exp_cfg_from_dict`) |
| `creation/` | Random trees and offspring: `treemo.py` (dispatch) and `creation.cu` (the operators) |
| `interpreters/` | `treemo_gpu/` (the Treemo kernel and rule extraction); `treemo_c/` and `treemo_py/` are the references the kernels are tested against |
| `rewards/` | Reward functions (`REWARDS` registry, `RewardSpec.zero_sum`) and `PayoffEngine` |
| `selection/` | Selection steps (`SELECTION_STEPS`) composed into a pipeline; `selection.cu` holds their kernels |
| `store/` | Population store (`store.publish`) and cross-population tournaments (`store.tournament`) |
| `sides/` | `random_baseline.py` — random-pool baseline with a fixed reference population |
| `loggers/` | `ExperimentLogger` (write side) and `run_io` (read side) of the run-directory format |
| `analysis/` | Ancestry visualization, tournament ratings, Treemo tree → graphviz |
| `tests/` | pytest suite; `tests/reference/` keeps the CPU implementations the GPU code is checked against |
| `docs/formats.md` | All on-disk formats + deferred design seams (islands, multi-reward, matchup cache) |
| `outputs/` | Per-run dirs, population store, tournaments (one timestamped subdir per run) |

## Treemo

`ExperimentConfig.interpreter` is `"treemo"`, the only interpreter (the field
stays so the compat keys of stored populations keep their values). The
language spec is [interpreters/treemo_py/README.md](interpreters/treemo_py/README.md);
`TreemoConfig` sets `max_step`, `tree_size` (random trees), `pass_mode` and
`first_mode` (how rules advance; see `config.py`).

On the device a program is a bit-packed tape in a heap
([core/cuda/tapes.cuh](core/cuda/tapes.cuh)), one warp runs each execution,
and rules are extracted for many programs at once; see
[interpreters/treemo_gpu/README.md](interpreters/treemo_gpu/README.md).

## Rewards

Selected via `ExperimentConfig.reward`: `blind | placeholder | quine_pressure`.

| Reward | File | Behaviour |
|---|---|---|
| `blind` | `rewards/blind_reward.py` | `game(a, b) = run(a, run(a, b)) == run(b, a)`; the reward is `game(a, b) - game(b, a)`. Zero-sum. |
| `placeholder` | `rewards/placeholder_reward.py` | Runs both programs on a fixed 0/1 input; whoever produces more output wins ±1, else 0. |
| `quine_pressure` | `rewards/quine_pressure_reward.py` | Each program runs on the other's code as input. Reward favors the program whose output more closely resembles *itself* (normalized LCS self-similarity), pushing toward quine-like behaviour. Fully symmetric/zero-sum. |

### Writing a reward

A reward scores a whole batch of matchups at once and returns one value per
matchup. Matchup `k` is program `m.a[k]` (whose reward it is) against `m.b[k]`:

```python
import numpy as np

def quine_pressure(m):
    imprint_a = m.run(m.a, m.b).similarity(m.a)  # A runs on B: how much of A is in the output
    imprint_b = m.run(m.b, m.a).similarity(m.b)
    return np.sign(imprint_a - imprint_b)
```

Register it in `rewards/base.py` as `RewardSpec(name, fn, zero_sum)`;
`zero_sum=True` promises `reward(b, a) == -reward(a, b)` and lets the engine
compute one orientation of each pair. Rewards combine: `def both(m): return
blind(m) + 2 * quine_pressure(m)` is a reward too. The operations
(`core/matchups.py`):

| Operation | Gives |
|---|---|
| `m.a`, `m.b` | the two players' programs (one sequence per matchup) |
| `m.const(seq)` | the same 0/1 sequence in every matchup |
| `m.run(code, inp)` | the output of `code[k]` run on `inp[k]`; outputs can be run or fed again |
| `m.memory(code, inp)` | the interpreter's second result (treemo: the code) |
| `t.concat(u)` | `t[k]` followed by `u[k]` |
| `t.length()`, `t.count(symbol)` | per-matchup numbers |
| `t.equals(u)`, `t.lcs(u)`, `t.similarity(ref)` | comparisons; `similarity` = LCS / `len(ref)`, 0 if either is empty |
| `m.map(fn, *args)` | any Python per matchup: `fn` gets lists (and array entries) — **host, slow, loud** |

Measurements are CuPy arrays on the device. Combine them with NumPy or CuPy
functions (`np.sign`, `np.where`, arithmetic work on CuPy arrays and stay on
the device); the engine casts the result to int32. Each distinct execution and
measurement is computed once per batch, so `run(m.b, m.a)` reuses the outputs
of `run(m.a, m.b)`.

Two things leave the GPU, and each prints a red **SLOW HOST PATH** banner
every time it happens: `m.map`, and the LCS of two tapes both longer than
16,384 bits (which runs in Python).

### Payoff engine

`rewards/payoff.py` provides `PayoffEngine`, the one place matchups are
scored: the evolution loop (`extend`, which reuses the old × old entries),
tournaments and tools (`block`, `matrix` for host genomes). A matrix is cut
into batches of about `PayoffConfig.gpu_batch_matchups` matchups;
`PayoffConfig.gpu_memory_mb` bounds one execution launch. For rewards that
are not zero-sum, both orientations of a pair go in the same batch so they
share their executions.

## Selection

Selection is an ordered pipeline of steps, each mapping the square payoff matrix
(device int32), **the population it is indexed by**, and the generation's random
numbers to surviving indices (`selection/base.py`): `select(payoff, pop, rng)`.
The population is passed because not every criterion is a function of the
payoff. To add a step, write it with CuPy (the NumPy API on the device) and
draw randomness from `rng` (`core/rng.py`: `uniform`, `integers`,
`permutation`), so it stays on the GPU and replays exactly on resume.

There are **two pipelines**, drawn from the same `SELECTION_STEPS` registry:

| Pipeline | When | Payoff argument |
|---|---|---|
| `RunConfig.pre_selection` | before `PayoffEngine.extend` | `None` — the matrix does not exist yet, so whatever this drops never costs a matchup |
| `RunConfig.selection` | after evaluation | the real matrix |

Only steps in `selection.base.PAYOFF_FREE_STEPS` may go in `pre_selection`;
`build_selection(..., pre=True)` rejects the rest rather than failing mid-run.

| Step | Config | Behaviour |
|---|---|---|
| `dedupe` | `DedupeStepConfig()` | Keep one individual per distinct genome (exact match, first occurrence). Default `pre_selection`, so duplicates are discovered for free. Selection by payoff alone cannot do this: identical genomes have identical payoff rows, so neither strictly dominates the other (`skim`) and their row-sums tie (`cap_top`) — duplicates otherwise survive as a block, and under an all-draws reward nothing is dominated at all, so `cap_top` ends up culling arbitrarily. Matching is on the genome, never on the payoff row, which is identical for *every* individual when the reward draws every matchup. |
| `skim` | `SkimStepConfig(n_rounds, fraction, n_accepted)` | Iterated elimination of strictly dominated strategies over the surviving columns (`selection/skim.py`). `fraction` controls what share of the dominated set is dropped per round; skimming stops early below `n_accepted`. |
| `cap_top` | `CapStepConfig(max_pop)` | Keep the `max_pop` best payoff row-sums; ties broken randomly. |
| `cap_random` | `CapStepConfig(kind="cap_random", max_pop)` | Uniform random downsample to `max_pop`. |
| `best_worst_random` | `BestWorstRandomStepConfig(n_best=1_000, n_worst=0, n_rand=0)` | Keep the `n_best` highest payoff row-sums, then the `n_worst` lowest among those left, then `n_rand` uniformly random remaining individuals without replacement. Scores use the full population entering this step; ties are broken randomly. Return sorted, unique indices; oversized quotas keep everyone. |
| `lexicase` | `LexicaseStepConfig(n_accepted=1_000)` | Select up to `n_accepted` unique survivors without replacement. Each pick shuffles opponent columns and filters to the exact highest payoff among remaining candidates on each case, breaking final ties uniformly. All original opponent columns remain available for every pick. Rows with identical payoff vectors are filtered as one group (no case can separate them), in one CUDA block. |
| `nash` | `NashStepConfig()` | Union of Nash-equilibrium supports via `nashpy` (`selection/nash_set.py`), **on the host** (loud banner). Support enumeration is exponential — small populations only. |
| `knn_score` | `KNNScoreStepConfig(n_accepted=1_000, k=10)` | Each program's single coordinate is its total payoff against the other programs (self-play excluded). Keep the `n_accepted` programs with the largest average absolute score difference to their `k` nearest other programs. Equal scores count as zero-distance neighbors; clamp `k` to the available neighbors and break selection ties randomly. Compute scores and neighborhoods once against the full population entering this step. |
| `mixture` | `MixtureStepConfig(n_best=1_000, n_good=1_000, n_good_novel=0, n_novelty=0, k=10)` | Keep `n_best` by highest payoff row-sum. From the next `n_good` by that same score, keep `n_good_novel` by the existing scalar `knn_score` criterion. Then keep `n_novelty` by the existing payoff-row `knn_novelty` criterion among everyone still unselected, including candidates passed over in the good pool. Both kNN rankings use the full input population and the same `k`; neither is recomputed after selecting a group. Require `n_good_novel <= n_good`; return sorted, unique indices, break ties randomly, and keep everyone when survivor quotas cover the population. |

For example, set `RunConfig.selection` to
`[BestWorstRandomStepConfig(n_best=150, n_worst=25, n_rand=25)]` to retain up to 200
individuals in those three groups.

To mix quality and novelty, use
`[MixtureStepConfig(n_best=150, n_good=500, n_good_novel=30, n_novelty=20, k=10)]`.
This retains up to 200 individuals: 150 best, 30 scalar-novel from the next 500
best, and 20 novel payoff profiles from the remainder. `n_good` is a candidate
pool size, not an additional survivor quota.

## Genetic operators (`creation/creation.cu`)

Each child is made by one GPU thread, following the operators of the CPU
version choice for choice (`tests/reference/creation_cpu.py`; the tests
compare their outcome distributions). For each offspring: with
`crossover_prob` (and at least two survivors) two distinct parents are drawn;
with `homoiconic_prob` the child is parent A's output run on B, unless that
output is empty; otherwise it is the configured crossover of A and B. The
rest are mutations of one survivor.

For `subtree`, each original non-root node contributes an independent trial with probability `mutation_rate`; the successful trials set the number of mutation events. Each event chooses one of four operations uniformly and samples its target uniformly from the current non-root nodes. Delete removes a whole subtree. Swap exchanges child branches between unrelated nodes (an empty branch is used for a childless node); if no second node is eligible, the event is skipped. Insert attaches a random subtree whose size is sampled from the current tree’s empirical subtree-size distribution, including the implicit root. Regenerate replaces the selected subtree with a random tree of the same size. The implicit root is never edited, so an empty tree stays empty. Insertions do not increase the event budget. Operations may leave a genome unchanged, for example regenerating a leaf. Standalone random creation uses `tree_size`.

Random trees shuffle `tree_size - 1` opens and closes and rotate after the
last minimum of the running height. That is not uniform over plane trees: a
tree's weight grows with the length of its first top-level subtree (the CPU
version did the same).

| Category | Options |
|---|---|
| Tree mutation (`tree_mutation_op`) | `subtree`: uniform delete, swap, insert, or regenerate; `leaf`: legacy leaf-only mutation |
| Tree crossover (`tree_crossover_op`) | `depth1`, `random_depth` |
| Homoiconic recombination | `output` (child = parent A's output when run on B) |

To add an operator: a `__device__` function in `creation/creation.cu` over the
child's bytes (it returns false when the child would outgrow its room), an op
code in the `create` kernel's switch, and its name in `creation/treemo.py`.

## Running experiments

One unified loop (`core/loop.py`) drives every experiment; a run is a
`RunConfig`. Presets reproduce the historical entry points:

| Preset | Description |
|---|---|
| `main` | Each generation adds `n_random` fresh + `n_offspring` bred individuals, drops duplicates, extends the payoff matrix incrementally, then applies `[skim, cap_top]`. |
| `evolution` | Fixed initial population (`n_init`), offspring only, `[skim, cap_top]`. |
| `random_skimmed` | Pure random injection (no breeding), `[skim, cap_random]`. |

All three get the default `pre_selection=[DedupeStepConfig()]`; pass
`pre_selection=[]` to run without it.

(`sides/random_baseline.py` remains a separate script: random pools scored
against a fixed reference population, no evolution.)

Every fresh run resolves and persists a `seed`. All randomness comes from
Philox4x32-10 keyed by that seed with counters (element, draw, generation,
call) (`core/rng.py`): a generation's draws depend only on the seed and the
generation, so checkpoints hold no generator state and the run is exactly
resumable: set `RunConfig.resume_from` to a run dir to continue it for
`n_iter` more generations (an interrupted run reproduces the uninterrupted one
bit-for-bit). Runs write
`outputs/<preset>/<timestamp>/` — config, metrics, per-generation
births/survivors deltas, optional payoff matrices (`payoff_every`); see
[docs/formats.md](docs/formats.md).

To restart from the survivors **after** a particular generation, set these
fields in `RunConfig` in `config.py`, then run `python main.py`:

```python
resume_from: Optional[str] = "outputs/main/20260929_164051"
resume_gen: Optional[int] = 399
```

This continues at generation **400** for `n_iter` additional generations, in the
same directory. It reconstructs the recorded population, reuses a compatible
payoff snapshot or recomputes the matrix, and preserves the ID high-water mark.
The generations that follow draw from the persisted seed as usual.
Before changing history it makes a complete backup in a hidden sibling directory
named `.<run>.before_gen_<generation>_<suffix>` (also printed and recorded in the
resume metadata). Metrics and population/payoff files after the selected generation
are removed from the active run, and its checkpoint is replaced before evolution
continues. Reconstruction or evaluation failures leave the original run unchanged.
Run this with the previous process stopped.

`resume_gen` defaults to `None`, preserving latest-checkpoint resume behavior.
Set it back to `None` after the restart; leaving `399` configured requests another
restart from 399 on every launch. `resume_gen` requires `resume_from` and a saved,
non-negative generation.

## Comparing methodologies (population store + tournaments)

Rewards are relative — an individual's score only means something against its
opponents, so different runs cannot be compared by their in-run payoffs.
Instead: publish each run's final population as a sample of its methodology,
then let stored populations play each other offline.

```bash
# publish a run's final population (or set RunConfig.publish_label to auto-publish)
uv run python -m store.publish outputs/main/<ts> --label quine-skim3

# create a tournament (fixes reward + interpreter compat), add populations
uv run python -m store.tournament create --dir outputs/store/tournaments/quine_v1 \
    --reward quine_pressure --from-pop <pop_id>
uv run python -m store.tournament add --dir outputs/store/tournaments/quine_v1 <pop_id> ...

# derive ratings from the stored payoff blocks (re-run any time)
uv run python -m store.tournament ratings --dir outputs/store/tournaments/quine_v1
```

Adding a population computes only the missing cross-population blocks, so
evaluating a new methodology against the existing pool is incremental. Only
populations with the same `compat_key` (interpreter + its settings) can meet;
ratings (mean payoff / win-rate, individual min/max for both, per-opponent
breakdown) are always relative to the member pool — the raw blocks are the
stable artifact. A tournament refuses new members when its reward's `zero_sum`
flag has changed since its blocks were computed. Composite runs
can seed generation 0 from stored populations via `RunConfig.seed_populations`.

## Analysis

- `analysis/ancestry_graph.py` / `analysis/ancestry_columns.py` — lineage
  visualizations over a run's births/survivors files.
- `analysis/tournament_ratings.py` — ratings from tournament blocks
  (Elo/Nash-averaging can be added over the same artifacts).
- `analysis/tree_viz.py` — Treemo tree (Dyck word of 0/1) → graphviz `digraph`.

## Todo's

- Matchup-result caching in `PayoffEngine.matrix` once matchup cost dominates
  (see the deferred-seams section of [docs/formats.md](docs/formats.md)).
- Maximal Loteries as selection

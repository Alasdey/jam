# jam

A jam trying to reach escape velocity.

Programs are short integer-encoded "genomes" interpreted by one of several toy
languages (Subleq, IconFracTran, Treemo). Pairs of programs are run against
each other, scored by a reward function into a zero-sum payoff matrix, and the
population evolves over generations via random injection, mutation, crossover,
and "homoiconic" recombination (using a program's own runtime output as
genetic material). Selection thins the population by iterated elimination of
dominated strategies (skim) and/or a Nash-equilibrium support subset.

## Setup

### 1. Python dependencies
```bash
uv sync
```

### 2. Subleq / Treemo C libraries
```bash
gcc -shared -fPIC -o interpreters/subleq/libsubleq.so interpreters/subleq/subleq.c
gcc -shared -fPIC -o interpreters/treemo_c/libtreemo.so interpreters/treemo_c/treemo.c
```

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
| `core/config_keys.py` | Config hashing (`compat_key`, `method_key`) and `exp_cfg_from_dict` |
| `main.py` | Thin dispatcher: `python main.py [preset]` → `core.loop.run` |
| `core/` | `types.py` (Program, Individual, Interpreter protocol) and `loop.py` (the unified generation loop, resume, checkpointing) |
| `creation/` | Program creators — random init, mutation, crossover, homoiconic recombination; `CREATORS` registry |
| `interpreters/` | Interpreter implementations (Subleq, IconFracTran, Treemo); `INTERPRETERS` registry |
| `rewards/` | Reward functions (`REWARDS` registry, `RewardSpec.zero_sum`) and `PayoffEngine` |
| `selection/` | Selection steps (`SELECTION_STEPS`: dedupe, skim, lexicase, cap_top, cap_random, nash) composed into a pipeline |
| `store/` | Population store (`store.publish`) and cross-population tournaments (`store.tournament`) |
| `sides/` | `random_baseline.py` — random-pool baseline with a fixed reference population |
| `loggers/` | `ExperimentLogger` (write side) and `run_io` (read side) of the run-directory format |
| `analysis/` | Ancestry visualization, tournament ratings, Treemo tree → graphviz |
| `tests/` | pytest suite (determinism, resume, reconstruction, store/tournament round-trips) |
| `docs/formats.md` | All on-disk formats + deferred design seams (islands, multi-reward, matchup cache) |
| `outputs/` | Per-run dirs, population store, tournaments (one timestamped subdir per run) |

## Interpreters

Selected via `ExperimentConfig.interpreter`: `subleq | iconfractran | treemo | treemo_py`.

| Interpreter | File(s) | Notes |
|---|---|---|
| `subleq` | `interpreters/subleq/` | Subtract-and-branch-if-less-or-equal-to-zero, via ctypes (`libsubleq.so`). Known to be flaky — see Todo's. |
| `iconfractran` | `interpreters/iconfractran/iconfractran.py` | FRACTRAN-like interpreter over integer programs. |
| `treemo` | `interpreters/treemo_c/` | Default. C implementation via ctypes (`libtreemo.so`); see [interpreters/treemo_c/README.md](interpreters/treemo_c/README.md) for build/CLI and compile-vs-exec tradeoffs. |
| `treemo_py` | `interpreters/treemo_py/` | Pure-Python reference implementation; see [interpreters/treemo_py/README.md](interpreters/treemo_py/README.md) for the full language spec (rule extraction, execution, termination). |

Programs/inputs are `list[int]` (Treemo trees are flattened to Dyck words of
0/1 bits). `TreemoConfig` exposes `pass_mode` and `first_mode` to control how
rules are advanced — see `config.py` for details.

## Rewards

Selected via `ExperimentConfig.reward`: `blind | placeholder | quine_pressure`.

| Reward | File | Behaviour |
|---|---|---|
| `blind` | `rewards/blind_reward.py` | Always returns 0 (no signal — used to test the loop without an evolutionary objective). |
| `placeholder` | `rewards/placeholder_reward.py` | Runs both programs against a fixed input; whoever produces more output wins ±1, else 0. |
| `quine_pressure` | `rewards/quine_pressure_reward.py` | Each program runs on the other's code as input. Reward favors the program whose output more closely resembles *itself* (normalized LCS self-similarity), pushing toward quine-like behaviour. Fully symmetric/zero-sum. |

`rewards/payoff.py` provides `PayoffEngine` — the single choke point through
which both the evolution loop and the tournament tool evaluate matchups.
`matrix(ref, pop)` computes a payoff block (in parallel when
`PayoffConfig.n_workers > 1`, with the worker pool reused across generations).
`PayoffConfig.chunksize` controls how many matchup tasks are submitted to a
worker at once; use `1` for the finest load balancing when matchup costs vary.
For chunks larger than one, the matchup list is shuffled reproducibly before
batching, using a separate RNG so scheduling does not affect evolution randomness.
`extend(payoff, old, new)` grows the square self-play matrix, deriving the
reverse block as `-A.T` when `RewardSpec.zero_sum` allows it.

## Selection

Selection is an ordered pipeline of steps, each mapping the square payoff matrix
**and the genomes it is indexed by** to surviving indices (`selection/base.py`).
Genomes are passed because not every criterion is a function of the payoff.

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
| `skim` | `SkimStepConfig(n_rounds, fraction, n_accepted)` | Iterated elimination of strictly dominated strategies (`selection/skim.py`; the non-`_fast` variant is kept for reference — it compares over all columns instead of the symmetric active set). `fraction` controls what share of the dominated set is dropped per round; skimming stops early below `n_accepted`. |
| `cap_top` | `CapStepConfig(max_pop)` | Keep the `max_pop` best payoff row-sums. |
| `cap_random` | `CapStepConfig(kind="cap_random", max_pop)` | Uniform random downsample to `max_pop`. |
| `lexicase` | `LexicaseStepConfig(n_accepted=1_000)` | Select up to `n_accepted` unique survivors without replacement. Each pick shuffles opponent columns and filters to the exact highest payoff among remaining candidates on each case, breaking final ties uniformly. All original opponent columns remain available for every pick. Uses bitsets over identical payoff rows when each case has at most eight distinct scores; otherwise falls back to array filtering. Both paths preserve individual tie probabilities and the original RNG stream. |
| `nash` | `NashStepConfig()` | Union of Nash-equilibrium supports via `nashpy` (`selection/nash_set.py`). Support enumeration is exponential — small populations only. |

## Genetic operators (`creation/genetics.py`)

For `subtree`, each original non-root node contributes an independent trial with probability `mutation_rate`; the successful trials set the number of mutation events. Each event chooses one of four operations uniformly and samples its target uniformly from the current non-root nodes. Delete removes a whole subtree. Swap exchanges child branches between unrelated nodes (an empty branch is used for a childless node); if no second node is eligible, the event is skipped. Insert attaches a random subtree whose size is sampled from the current tree’s empirical subtree-size distribution, including the implicit root. Regenerate replaces the selected subtree with a random tree of the same size. The implicit root is never edited, so an empty tree stays empty. Insertions do not increase the event budget. Operations may leave a genome unchanged, for example regenerating a leaf. Standalone random creation is separate and still uses `tree_size`. SUBLEQ and other integer programs retain independent per-integer mutation probabilities.

| Category | Options |
|---|---|
| Code mutation (`code_mutation_op`) | `uniform` (replace gene w.p. `mutation_rate`), `creep` (nudge gene by ±delta) |
| Code crossover (`code_crossover_op`) | `single_point`, `two_point`, `uniform` |
| Tree mutation (`tree_mutation_op`) | `subtree`: uniform delete, swap, insert, or regenerate; `leaf`: legacy leaf-only mutation |
| Tree crossover (`tree_crossover_op`) | `depth1`, `random_depth` |
| Homoiconic recombination | `output` (child = parent A's output when run on B), `memory` (child = resulting memory/state) |

`creation/offspring.py` combines these: each offspring is produced via
crossover (with `homoiconic_prob` chance of trying homoiconic recombination
first) or mutation of a single survivor, per `GeneticsConfig`.

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

Every fresh run resolves and persists an RNG `seed`, checkpoints each
generation, and is exactly resumable: set `RunConfig.resume_from` to a run dir
to continue it for `n_iter` more generations (RNG state is restored, so an
interrupted run reproduces the uninterrupted one bit-for-bit). Runs write
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
It does **not** seed or restore either RNG; exact historical replay is not required.
Before changing history it makes a complete backup in a hidden sibling directory
named `.<run>.before_gen_<generation>_<suffix>` (also printed and recorded in the
resume metadata). Metrics and population/payoff files after the selected generation
are removed from the active run, and its checkpoint is replaced before evolution
continues. Reconstruction or evaluation failures leave the original run unchanged.
Run this with the previous process stopped.

`resume_gen` defaults to `None`, preserving latest-checkpoint resume behavior.
Set it back to `None` after the restart; leaving `399` configured requests another
restart from 399 on every launch. `resume_gen` requires `resume_from` and a saved,
non-negative generation. No extra per-generation RNG checkpoints are stored.

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
    --reward quine_pressure --from-pop <pop_id> --workers 8
uv run python -m store.tournament add --dir outputs/store/tournaments/quine_v1 <pop_id> ...

# derive ratings from the stored payoff blocks (re-run any time)
uv run python -m store.tournament ratings --dir outputs/store/tournaments/quine_v1
```

Adding a population computes only the missing cross-population blocks, so
evaluating a new methodology against the existing pool is incremental. Only
populations with the same `compat_key` (interpreter + its settings) can meet;
ratings (mean payoff / win-rate, individual min/max for both, per-opponent
breakdown) are always relative to the member pool — the raw blocks are the
stable artifact. Composite runs
can seed generation 0 from stored populations via `RunConfig.seed_populations`.

## Analysis

- `analysis/ancestry_graph.py` / `analysis/ancestry_columns.py` — lineage
  visualizations over a run's births/survivors files.
- `analysis/tournament_ratings.py` — ratings from tournament blocks
  (Elo/Nash-averaging can be added over the same artifacts).
- `analysis/tree_viz.py` — Treemo tree (Dyck word of 0/1) → graphviz `digraph`.

## Todo's

- Subleq is broken in some way — payoffs computed with it produce matrices
  that don't make sense.
- Investigate whether a module-level compile cache is worth adding for
  `treemo_c` at low `max_step` regimes (see
  [interpreters/treemo_c/README.md](interpreters/treemo_c/README.md)).
- Matchup-result caching in `PayoffEngine.matrix` once matchup cost dominates
  (see the deferred-seams section of [docs/formats.md](docs/formats.md)).
- Maximal Loteries as selection

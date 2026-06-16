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
uv run python main.py
```

## Project layout

| Path | Role |
|---|---|
| `config.py` | All dataclass configs (experiment, genetics, payoff, and per-entrypoint configs) |
| `main.py` | Main evolutionary loop: random injection + offspring, payoff matrix, skim, max-pop cap |
| `creation/` | Program creators — random init, mutation, crossover, homoiconic recombination, per-interpreter |
| `interpreters/` | Interpreter implementations (Subleq, IconFracTran, Treemo) and the `make_interpreter` factory |
| `rewards/` | Reward functions and payoff-matrix computation |
| `selection/` | Skim (iterated elimination of dominated strategies) and Nash-subset selection |
| `sides/` | Alternative/experimental entrypoints (random baseline, random+skim, full evolution loop, homoiconic round-robin) |
| `loggers/` | `ExperimentLogger` — writes config/meta/metrics/population files per run |
| `analysis/` | Visualization helpers (e.g. Treemo tree → graphviz) |
| `outputs/` | Per-run logs and population dumps (one timestamped subdir per run) |

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

`rewards/payoff.py` computes the full payoff matrix for a reference population
vs. a working population (`compute_payoff_matrix`), optionally in parallel via
`ProcessPoolExecutor` (`PayoffConfig.n_workers > 1`).

## Selection

- `selection/skim.py` — iterated elimination of strictly dominated rows.
  `iterated_elimination_strictly_dominated_rows_fast` is the vectorized version
  used by default; the non-`_fast` version is kept for reference/validation.
  `skim_fraction` (in the main/evolution configs) controls what fraction of the
  dominated set is actually dropped per round (1.0 = drop all, 0.0 = keep all).
- `selection/nash_set.py` — `compute_nash_equilibrium` / `compute_nash_subset`
  use `nashpy` to find Nash equilibria of the zero-sum payoff matrix and return
  the union of equilibrium supports.

## Genetic operators (`creation/genetics.py`)

| Category | Options |
|---|---|
| Code mutation (`code_mutation_op`) | `uniform` (replace gene w.p. `mutation_rate`), `creep` (nudge gene by ±delta) |
| Code crossover (`code_crossover_op`) | `single_point`, `two_point`, `uniform` |
| Tree mutation (`tree_mutation_op`) | `leaf`, `subtree` |
| Tree crossover (`tree_crossover_op`) | `depth1`, `random_depth` |
| Homoiconic recombination | `output` (child = parent A's output when run on B), `memory` (child = resulting memory/state) |

`creation/offspring.py` combines these: each offspring is produced via
crossover (with `homoiconic_prob` chance of trying homoiconic recombination
first) or mutation of a single survivor, per `GeneticsConfig`.

## Entrypoints

| Script | Config | Description |
|---|---|---|
| `main.py` | `MainConfig` | Main loop: each generation adds `n_random` fresh + `n_offspring` bred individuals, computes the payoff matrix incrementally, skims dominated rows (`n_skim` rounds, `skim_fraction`), and caps the population at `max_pop` by top payoff sum. |
| `sides/evolution.py` | `EvolutionConfig` | Generalized evolutionary loop with pluggable selection (`skim_fast`, `skim_slow`, `nash_subset`, `none`). |
| `sides/random_skimmed.py` | `RandomSkimmedConfig` | Population grows by pure random injection only (no breeding), with periodic skimming. |
| `sides/random_baseline.py` | `RandomBaselineConfig` | Samples random pools and tracks the best payoff seen, with no selection/evolution — a baseline for comparison. |
| `sides/homoiconic.py` | n/a (hardcoded paths) | Loads a saved population and round-robin scores candidate programs against it. |

All entrypoints write to `outputs/<name>/<timestamp>/` via `ExperimentLogger`
(`config.json`, `meta.json`, `metrics.jsonl`, `work_pop.json`, `ref_pop.json`).

## Analysis

`analysis/tree_viz.py` turns a Treemo tree (Dyck word of 0/1) into a graphviz
`digraph` string for visualization.

Also in the outputs logs this could help:
```regex
(.*"pop_size": 1000,.*\n)*.*"pop_size": 1,.*"random": 1\}
```

## Todo's

- Unit testing / sanity checks for interpreters, rewards, and selection are
  still missing.
- Subleq is broken in some way — payoffs computed with it produce matrices
  that don't make sense.
- Investigate whether a module-level compile cache is worth adding for
  `treemo_c` at low `max_step` regimes (see
  [interpreters/treemo_c/README.md](interpreters/treemo_c/README.md)).

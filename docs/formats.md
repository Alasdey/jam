# On-disk formats

All stable artifacts are versioned JSON / JSONL / npz with a `format_version`
field. `checkpoint.pkl` is the one exception: a volatile pickle whose layout
may change between versions — everything needed long-term lives in the stable
files.

## Run directory — `outputs/<preset>/<timestamp>/`

Written by `loggers/experiment_logger.py`, read by `loggers/run_io.py`.

```
config.json                  full RunConfig (dataclass asdict), with the RESOLVED seed
config_resumed_<ts>.json     config of each resume invocation (original config.json is kept)
meta.json                    {format_version, start_time, start_ts, git_hash,
                              seeded_provenance?}   (meta_resumed_<ts>.json per resume)
metrics.jsonl                one JSON record per generation (appended across resumes)
populations/
  births_XXXXXX.jsonl        newborn individuals persisted at generation XXXXXX, one per line:
                             {"id", "genome", "method", "parents", "born_gen"}
  survivors_XXXXXX.json      {"gen", "ids": [...], "scores": [...]}
                             ids = population after selection, in matrix order;
                             scores[i] = post-selection payoff row-sum of ids[i]
payoffs/
  payoff_XXXXXX.npz          arrays: payoff (int32, square, post-selection),
                             ids (int64, aligned with rows/cols)
                             written every RunConfig.payoff_every generations + final gen
checkpoint.pkl               volatile resume state, atomically replaced each generation:
                             gen, next_id, pop_ids, payoff, py/np RNG states
```

Reconstruction invariant: the population surviving generation G is
`survivors_G["ids"]` resolved against the union of all `births_*.jsonl` with
generation <= G (`run_io.reconstruct_population`). With the default
`log_all_births=False` only newborns that survive their birth generation are
persisted — sufficient because an individual removed from the population never
returns. Set `log_all_births=True` to also keep immediately-culled newborns
(full creation-method statistics at the cost of much larger births files).

- Individual `method`: `"random" | "mutate" | "crossover" | "homoiconic" | "seeded"`.
- Individual ids are unique within one run (monotonic across resumes;
  the checkpoint carries `next_id`).
- `metrics.jsonl` per-generation fields: `gen`, `t`, `create_s`, `payoff_s`,
  `select_s`, `pop_size`, `n_added`, `n_added_by_method`, `n_removed`,
  `n_removed_by_step` (one entry per selection step, in pipeline order),
  `n_survived_new`, `n_survived_new_by_method`, and `payoff_*` / `payout_*`
  stats (row-sum mean/std/min/max, post- and pre-selection respectively).
- `seeded_provenance` (composite runs): list of `"<pop_id>/<original_id>"`;
  entry i describes the seeded individual with local id i.

## Config keys

Defined in `config.py`:

- `compat_key(experiment_dict)` — sha256[:12] of `{interpreter, <its
  sub-config>}`. Execution semantics only: two populations can play each other
  iff their compat_keys match. Reward, genetics, and code bounds are excluded
  (they shape creation, not play). `treemo` and `treemo_py` are deliberately
  distinct until the implementations are proven equivalent.
- `method_key(run_config_dict)` — sha256[:12] of the RunConfig minus run
  identity (`seed`, `out_dir`, `resume_from`, `publish_label`, `store_dir`).
  Samples of the same methodology across seeds share this key.

## Population store — `outputs/store/populations/<pop_id>/`

Written/read by `store/population_store.py`
(CLI: `python -m store.publish <run_dir> --label <label> [--gen N]`).

`pop_id = <label>__<YYYYMMDD_HHMMSS_us>__<compat_key[:8]>`

```
manifest.json        {format_version, pop_id, label, created, source_run, source_gen,
                      n_individuals, compat_key, method_key, seed, git_hash,
                      run_config: {...full RunConfig dict...}}
individuals.jsonl    one Individual record per line (same schema as births files)
```

Composite seeding: `RunConfig.seed_populations = [pop_id, ...]` injects every
individual of those store populations at generation 0 with `method="seeded"`
and records `seeded_provenance` in `meta.json`. Seeded individuals join the
breeding pool once they survive into generation 1.

## Tournament — `outputs/store/tournaments/<tid>/`

Written by `store/tournament.py`, ratings derived by
`analysis/tournament_ratings.py`.

```
tournament.json      {format_version, tournament_id, reward, zero_sum, compat_key,
                      store_dir, experiment_config, members: [pop_id, ...]}
blocks/
  <popA>__vs__<popB>.npz    arrays: payoff (int32, rows=popA individuals,
                            cols=popB individuals), row_ids, col_ids (int64)
  <popA>__vs__<popB>.json   sidecar: {row_pop, col_pop, reward, n_workers,
                            elapsed_s, computed_at}
ratings.json         derived; regenerate any time with `store.tournament ratings`
```

- One reward + one compat_key per tournament; adding an incompatible
  population fails loudly. The reward need not be the one the populations
  were evolved under.
- Zero-sum rewards store only the canonical `sorted(popA, popB)` orientation;
  the reverse block is `-A.T` (`analysis.tournament_ratings.load_block`
  handles this). Non-zero-sum rewards would store both orientations.
- `add` is incremental: only missing blocks (new member vs. each existing
  member) are computed; existing block files are never rewritten.
- Cross blocks only by default; `--self` also computes self-play blocks
  (excluded from ratings either way).
- `ratings.json`: per population `mean_payoff`, `win_rate`, `n_matchups`
  and a `per_opponent` breakdown, ranked by mean_payoff. Ratings are relative
  to the member pool — the raw blocks are the ground truth; re-derive after
  every membership change.

## Deferred design seams (documented, not built)

- **Islands / multi-pool runs**: `core.loop._step` is a pure function over an
  explicit `RunState`; a multi-pool driver holds N states, steps each, and
  migrates individuals between them (the store provenance format already
  describes migrants). No changes to `_step` required.
- **Multi-reward optimization**: evaluate one payoff block per reward and
  column-stack them; skim already handles rectangular matrices. The one
  revisit is `PayoffEngine.extend`, which assumes a single square matrix.
  `RewardSpec.zero_sum` already marks where `-A.T` shortcuts stop applying.
- **Matchup-result cache**: key `(hash(code_a), hash(code_b), reward,
  compat_key)` inside `PayoffEngine.matrix` — the single choke point through
  which the loop and the tournament evaluate matchups. Worth building when
  matchup cost dominates (heavy rewards, composite runs replaying stored
  individuals).
- **Elo / Nash-averaging ratings**: additional functions in
  `analysis/tournament_ratings.py` over the same stored blocks; the block
  format needs no change.
- **Cross-interpreter comparison**: impossible at the reward level (programs
  from different languages cannot play each other). If ever needed, it will be
  a separate layer of interpreter-agnostic absolute descriptors (output
  statistics, compression-based complexity, self-replication measures), not a
  tournament feature.

"""
Cross-population tournaments over the population store.

A tournament fixes one reward and one interpreter compat_key; member
populations play full cross-population matchups. Raw payoff blocks are the
stable artifact; ratings are derived from them on demand.

  <tournament_dir>/
    tournament.json                    — reward, compat_key, members
    blocks/<popA>__vs__<popB>.npz      — payoff (int32), row_ids, col_ids
    blocks/<popA>__vs__<popB>.json     — sidecar: timing + provenance
    ratings.json                       — derived (store.tournament ratings)

For zero-sum rewards only the canonical sorted(popA, popB) orientation is
stored; the reverse is -A.T. Non-zero-sum rewards store both orientations.
Adding a population computes only the missing blocks (new vs. each member).

Usage:
    python -m store.tournament create --dir <d> --reward <name> --from-pop <pop_id>
                                      [--store DIR] [--workers N] [--self]
    python -m store.tournament add    --dir <d> <pop_id> [...] [--workers N] [--self]
    python -m store.tournament ratings --dir <d>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from core.config_keys import exp_cfg_from_dict
from rewards.base import REWARDS
from rewards.payoff import PayoffEngine
from store.population_store import load


def _doc_path(tournament_dir: str) -> Path:
    return Path(tournament_dir) / "tournament.json"


def load_tournament(tournament_dir: str) -> dict:
    with open(_doc_path(tournament_dir)) as f:
        return json.load(f)


def _pairs_to_store(pop_a: str, pop_b: str, zero_sum: bool) -> list[tuple[str, str]]:
    if pop_a == pop_b or zero_sum:
        return [tuple(sorted((pop_a, pop_b)))]
    return [(pop_a, pop_b), (pop_b, pop_a)]


def create_tournament(
    tournament_dir: str,
    reward_name: str,
    from_pop: str,
    store_dir: str,
    n_workers: int = 1,
    include_self: bool = False,
) -> None:
    spec = REWARDS[reward_name]
    manifest, _ = load(store_dir, from_pop)

    tdir = Path(tournament_dir)
    tdir.mkdir(parents=True)  # an existing tournament dir is an error
    (tdir / "blocks").mkdir()

    doc = {
        "format_version": 1,
        "tournament_id": tdir.name,
        "reward": reward_name,
        "zero_sum": spec.zero_sum,
        "compat_key": manifest["compat_key"],
        "store_dir": str(store_dir),
        # interpreter settings for matchups; identical across members by compat_key
        "experiment_config": manifest["run_config"]["experiment"],
        "members": [],
    }
    with open(_doc_path(tournament_dir), "w") as f:
        json.dump(doc, f, indent=2)

    add_populations(tournament_dir, [from_pop], n_workers, include_self)


def add_populations(
    tournament_dir: str,
    pop_ids: list[str],
    n_workers: int = 1,
    include_self: bool = False,
) -> None:
    doc = load_tournament(tournament_dir)
    spec = REWARDS[doc["reward"]]
    store_dir = doc["store_dir"]
    blocks_dir = Path(tournament_dir) / "blocks"

    exp_cfg = exp_cfg_from_dict(doc["experiment_config"])
    exp_cfg.reward = doc["reward"]
    exp_cfg.payoff.n_workers = n_workers

    populations: dict[str, tuple[list[int], list]] = {}

    def ids_and_genomes(pop_id: str):
        if pop_id not in populations:
            manifest, individuals = load(store_dir, pop_id)
            if manifest["compat_key"] != doc["compat_key"]:
                raise ValueError(
                    f"{pop_id} has compat_key {manifest['compat_key']}, "
                    f"tournament requires {doc['compat_key']}"
                )
            populations[pop_id] = (
                [r["id"] for r in individuals],
                [r["genome"] for r in individuals],
            )
        return populations[pop_id]

    def compute_block(engine: PayoffEngine, row_pop: str, col_pop: str) -> None:
        path = blocks_dir / f"{row_pop}__vs__{col_pop}.npz"
        if path.exists():
            return
        row_ids, row_genomes = ids_and_genomes(row_pop)
        col_ids, col_genomes = ids_and_genomes(col_pop)
        t0 = time.time()
        payoff = engine.matrix(row_genomes, col_genomes)
        elapsed = time.time() - t0
        np.savez_compressed(
            path,
            payoff=payoff.astype(np.int32),
            row_ids=np.array(row_ids, dtype=np.int64),
            col_ids=np.array(col_ids, dtype=np.int64),
        )
        sidecar = {
            "format_version": 1,
            "row_pop": row_pop,
            "col_pop": col_pop,
            "reward": doc["reward"],
            "n_workers": n_workers,
            "elapsed_s": round(elapsed, 4),
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(sidecar, f, indent=2)
        print(f"computed {row_pop} vs {col_pop} ({payoff.shape[0]}x{payoff.shape[1]}, {elapsed:.2f}s)")

    with PayoffEngine(exp_cfg, spec) as engine:
        for new_pop in pop_ids:
            if new_pop in doc["members"]:
                raise ValueError(f"{new_pop} is already a tournament member")
            ids_and_genomes(new_pop)  # compat check before any compute
            for member in doc["members"]:
                for row_pop, col_pop in _pairs_to_store(member, new_pop, spec.zero_sum):
                    compute_block(engine, row_pop, col_pop)
            if include_self:
                compute_block(engine, new_pop, new_pop)
            doc["members"].append(new_pop)

    with open(_doc_path(tournament_dir), "w") as f:
        json.dump(doc, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="create a tournament seeded with one population")
    c.add_argument("--dir", required=True)
    c.add_argument("--reward", required=True, choices=sorted(REWARDS))
    c.add_argument("--from-pop", required=True)
    c.add_argument("--store", default="outputs/store/populations")
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--self", action="store_true", dest="include_self",
                   help="also compute self-play blocks")

    a = sub.add_parser("add", help="add populations, computing only missing blocks")
    a.add_argument("pop_ids", nargs="+")
    a.add_argument("--dir", required=True)
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--self", action="store_true", dest="include_self")

    r = sub.add_parser("ratings", help="derive ratings.json from the stored blocks")
    r.add_argument("--dir", required=True)

    args = parser.parse_args()
    if args.cmd == "create":
        create_tournament(args.dir, args.reward, args.from_pop, args.store,
                          args.workers, args.include_self)
    elif args.cmd == "add":
        add_populations(args.dir, args.pop_ids, args.workers, args.include_self)
    elif args.cmd == "ratings":
        from analysis.tournament_ratings import write_ratings

        ratings = write_ratings(args.dir)
        header = f"{'pop_id':<50} {'mean_payoff':>12} {'win_rate':>9} {'n_matchups':>11}"
        print(header)
        print("-" * len(header))
        for pop_id, r in ratings["populations"].items():
            print(f"{pop_id:<50} {r['mean_payoff']:>12.4f} {r['win_rate']:>9.4f} {r['n_matchups']:>11}")


if __name__ == "__main__":
    main()

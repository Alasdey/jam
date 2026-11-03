"""
Selection methods, several seeds each, in one tournament.

Runs every selection method of the `selections` tournament (plus mixture)
once per seed with the main preset, publishes each run's final population,
adds them all to a tournament, and summarizes the ratings per method across
seeds.

    python -m sides.selection_tournament                         # 5 methods x seeds 1 2 3, 1,000 generations
    python -m sides.selection_tournament --seeds 4 5 6           # more seeds: added to the same tournament
    python -m sides.selection_tournament --methods mixture lexicase --n-iter 500
    python -m sides.selection_tournament --with-old              # also the four original populations

Nothing is resumed: a run dir left by an interrupted run stops the script
with an error (delete it to run that seed again). Finished runs are reused: a
run's pop_id is kept in <runs-dir>/<method>_s<seed>/published.txt, it is not
run or published again, and only populations not yet in the tournament are
added (only missing blocks are computed).

By default the tournament is a new one, outputs/store/tournaments/selections_v2:
the original `selections` tournament was computed with an earlier definition
of blind, so mixing its blocks with new ones would compare different rewards.
--tournament points at any tournament dir (created if missing).
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.tournament_ratings import load_block, write_ratings
from config import (
    BestWorstRandomStepConfig, KNNNoveltyStepConfig, KNNScoreStepConfig, LexicaseStepConfig, MixtureStepConfig,
    RunConfig,
)
from core.loop import run
from store.population_store import publish
from store.tournament import add_populations, create_tournament, load_tournament

# The selection of each run behind the `selections` tournament, and mixture.
METHODS = {
    "lexicase": lambda: [LexicaseStepConfig(n_accepted=200)],
    "knn_novelty": lambda: [KNNNoveltyStepConfig(n_accepted=200, k=10)],
    "knn_score": lambda: [KNNScoreStepConfig(n_accepted=200, k=10)],
    "best_worst_random": lambda: [BestWorstRandomStepConfig(n_best=150, n_worst=20, n_rand=30)],
    "mixture": lambda: [MixtureStepConfig(n_best=150, n_good=500, n_good_novel=30, n_novelty=20, k=10)],
}

# The populations of the original `selections` tournament (evolved on the CPU under an earlier blind).
OLD_POPULATIONS = {
    "treemo-lexicase__20261005_133609_224095__d09bf6ed": "lexicase (old)",
    "treemo-knnnovelty__20261005_133752_801457__d09bf6ed": "knn_novelty (old)",
    "treemo-knnscalar__20261006_120702_705843__d09bf6ed": "knn_score (old)",
    "best_worst_random__20261010_201727_208765__d09bf6ed": "best_worst_random (old)",
}


def run_and_publish(method: str, seed: int, n_iter: int, runs_dir: Path, store_dir: str) -> str:
    """The pop_id of this (method, seed) run's final population: a fresh run, or the published one."""
    run_dir = runs_dir / f"{method}_s{seed}"
    published = run_dir / "published.txt"
    if published.exists():
        return published.read_text().strip()
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} was left by an unfinished run; delete it to run {method} seed {seed} again")

    print(f"\n=== {method}, seed {seed}: {n_iter} generations into {run_dir}")
    t0 = time.time()
    run(RunConfig(seed=seed, n_iter=n_iter, selection=METHODS[method](), out_dir=str(run_dir),
                  publish_label=None, store_dir=store_dir))
    print(f"=== {method}, seed {seed}: {time.time() - t0:.0f} s")

    pop_id = publish(store_dir, f"sel-{method}-s{seed}", str(run_dir))
    published.write_text(pop_id + "\n")
    print(f"=== published {pop_id}")
    return pop_id


def add_to_tournament(tournament_dir: Path, store_dir: str, reward: str, pop_ids: list[str]) -> None:
    if not (tournament_dir / "tournament.json").exists():
        create_tournament(str(tournament_dir), reward, pop_ids[0], store_dir)
    members = set(load_tournament(str(tournament_dir))["members"])
    missing = [p for p in pop_ids if p not in members]
    if missing:
        add_populations(str(tournament_dir), missing)


def summarize(tournament_dir: Path, labels: dict[str, str]) -> str:
    """Ratings per method across seeds, and the method x method mean payoff, as markdown tables."""
    ratings = write_ratings(str(tournament_dir))["populations"]
    zero_sum = load_tournament(str(tournament_dir))["zero_sum"]
    pops = [p for p in ratings if p in labels]
    methods = sorted({labels[p] for p in pops}, key=lambda m: -np.mean(
        [ratings[p]["mean_payoff"] for p in pops if labels[p] == m]))

    lines = ["| Method | Runs | Mean payoff (± sd over runs) | Win rate (± sd) | Per run |", "|---|---|---|---|---|"]
    for m in methods:
        mine = [p for p in pops if labels[p] == m]
        payoff = np.array([ratings[p]["mean_payoff"] for p in mine])
        wins = np.array([ratings[p]["win_rate"] for p in mine])
        per_run = " ".join(f"{x:+.3f}" for x in payoff)
        lines.append(f"| {m} | {len(mine)} | {payoff.mean():+.3f} ± {payoff.std():.3f} | "
                     f"{wins.mean():.3f} ± {wins.std():.3f} | {per_run} |")

    # Mean payoff of each method's populations against each method's (other) populations.
    cells = defaultdict(list)
    for p in pops:
        for q in pops:
            if p != q:
                cells[labels[p], labels[q]].append(load_block(str(tournament_dir), p, q, zero_sum).mean())
    lines += ["", "Mean payoff, row method against column method (other runs of the same method on the diagonal):",
              "", "| | " + " | ".join(methods) + " |", "|---" * (len(methods) + 1) + "|"]
    for a in methods:
        row = [f"{np.mean(cells[a, b]):+.3f}" if cells[a, b] else "—" for b in methods]
        lines.append(f"| {a} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n-iter", type=int, default=1_000, help="generations per run")
    parser.add_argument("--runs-dir", default="outputs/selection_runs")
    parser.add_argument("--store", default="outputs/store/populations")
    parser.add_argument("--tournament", default="outputs/store/tournaments/selections_v2")
    parser.add_argument("--with-old", action="store_true",
                        help="also add the four populations of the original selections tournament")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    labels: dict[str, str] = {}
    for method in args.methods:
        for seed in args.seeds:
            labels[run_and_publish(method, seed, args.n_iter, runs_dir, args.store)] = method
    if args.with_old:
        labels.update(OLD_POPULATIONS)

    tournament_dir = Path(args.tournament)
    reward = RunConfig().experiment.reward
    add_to_tournament(tournament_dir, args.store, reward, list(labels))

    # Label every member this campaign knows, including those of earlier invocations.
    for path in runs_dir.glob("*_s*/published.txt"):
        method = path.parent.name.rsplit("_s", 1)[0]
        labels.setdefault(path.read_text().strip(), method)
    members = set(load_tournament(str(tournament_dir))["members"])
    labels = {p: m for p, m in labels.items() if p in members}
    if args.with_old or any(p in members for p in OLD_POPULATIONS):
        labels.update({p: m for p, m in OLD_POPULATIONS.items() if p in members})

    report = summarize(tournament_dir, labels)
    (tournament_dir / "summary.md").write_text(report + "\n")
    print("\n" + report)
    print(f"\n(also in {tournament_dir / 'summary.md'})")


if __name__ == "__main__":
    main()

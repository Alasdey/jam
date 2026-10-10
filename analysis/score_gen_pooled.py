"""
Score each generation's survivors against one fixed pool of codes drawn from
the whole run, and plot score vs. generation.

The pool is a uniform sample without replacement of distinct persisted
genomes across the run. Identical code under different birth IDs counts once;
survival across generations adds no sampling weight. For each generation, a
`--proportion` of the survivors is sampled and scored as its mean reward
against the pool (row = evaluated code, column = pool code). Codes that are
both in the pool and evaluated are not special-cased.

Output: <run_dir>/score_gen_pooled.svg — a point cloud of every evaluated
score by generation, with the per-generation mean drawn as a line.

Usage:
    python -m analysis.score_gen_pooled <run_id> --n-pool N --proportion P [--workers W] [--seed S]

<run_id> is either a run dir (outputs/main/20260910_181415) or its bare id
(20260910_181415), looked up under outputs/*/.
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.config_keys import exp_cfg_from_dict
from loggers.run_io import last_gen, load_births, load_survivors
from rewards.base import REWARDS
from rewards.payoff import PayoffEngine


def resolve_run_dir(run_id: str) -> Path:
    if Path(run_id).is_dir():
        return Path(run_id)
    matches = list(Path("outputs").glob(f"*/{run_id}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one run dir for {run_id} under outputs/*/, found {matches}")
    return matches[0]


def generations(run_dir: Path) -> list[int]:
    return sorted(
        int(p.stem.rsplit("_", 1)[-1])
        for p in (run_dir / "populations").glob("survivors_*.json")
    )


def sample_unique_pool(births: dict[int, dict], n_pool: int, rng: random.Random) -> list[list[int]]:
    """Sample distinct code contents, independent of birth multiplicity."""
    unique = list(dict.fromkeys(tuple(births[i]["genome"]) for i in sorted(births)))
    if not 1 <= n_pool <= len(unique):
        raise ValueError(
            f"n_pool must be between 1 and {len(unique)} distinct persisted programs; "
            f"got {n_pool}"
        )
    return [list(genome) for genome in rng.sample(unique, n_pool)]


def score_gen_pooled(
    run_dir: Path, n_pool: int, proportion: float, n_workers: int, seed: int,
    backend: str = "cpu",
) -> tuple[list[int], list[float]]:
    rng = random.Random(seed)
    with open(run_dir / "config.json") as f:
        exp_dict = json.load(f)["experiment"]
    exp_cfg = exp_cfg_from_dict(exp_dict)
    exp_cfg.payoff.n_workers = n_workers
    exp_cfg.payoff.backend = backend
    spec = REWARDS[exp_cfg.reward]

    births = load_births(str(run_dir), up_to_gen=last_gen(str(run_dir)))
    pool = sample_unique_pool(births, n_pool, rng)

    # Draw every generation's sample up front so the total work is known for the ETA.
    samples: list[tuple[int, list[int]]] = []
    for gen in generations(run_dir):
        ids = load_survivors(str(run_dir), gen)["ids"]
        k = round(proportion * len(ids))
        if k > 0:
            samples.append((gen, rng.sample(ids, k)))
    total = sum(len(ids) for _, ids in samples)

    gens: list[int] = []
    scores: list[float] = []
    score_by_code: dict[tuple[int, ...], float] = {}
    t0 = time.time()
    with PayoffEngine(exp_cfg, spec) as engine:
        for gen, ids in samples:
            keys = [tuple(births[i]["genome"]) for i in ids]
            uncached = list(dict.fromkeys(key for key in keys if key not in score_by_code))
            if uncached:
                new_scores = engine.matrix([list(key) for key in uncached], pool).mean(axis=1)
                score_by_code.update(zip(uncached, new_scores.tolist()))
            gens.extend([gen] * len(ids))
            scores.extend(score_by_code[key] for key in keys)
            print_progress(gen, len(scores), total, time.time() - t0)
    print()
    return gens, scores


def _fmt_s(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def print_progress(gen: int, done: int, total: int, elapsed: float) -> None:
    # Every matchup costs one evaluated code x the whole pool, so time scales with codes scored.
    remaining = elapsed / done * (total - done)
    print(
        f"\rgen {gen} | {done}/{total} codes ({100 * done / total:.1f}%) "
        f"| elapsed {_fmt_s(elapsed)} | remaining ~{_fmt_s(remaining)}",
        end="",
        flush=True,
    )


def plot(gens: list[int], scores: list[float], out_path: Path, title: str) -> None:
    g = np.array(gens)
    s = np.array(scores)
    uniq = np.unique(g)
    means = np.array([s[g == u].mean() for u in uniq])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(g, s, s=4, alpha=0.25, color="#4a78b5", linewidths=0, label="evaluated code")
    ax.plot(uniq, means, color="#d1603d", linewidth=1.5, label="generation mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean reward vs. pool")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="Run dir or bare run id, e.g. 20260910_181415")
    parser.add_argument("--n-pool", type=int, required=True, help="Number of codes in the pool")
    parser.add_argument("--proportion", type=float, required=True, help="Fraction of each generation's survivors to score")
    parser.add_argument("--workers", type=int, default=1, help="Parallel processes for matchup evaluation")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu",
                        help="Payoff backend; cuda needs a GPU (treemo only)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()



    run_dir = resolve_run_dir(args.run_id)
    gens, scores = score_gen_pooled(run_dir, args.n_pool, args.proportion, args.workers, args.seed,
                                    args.backend)
    out_path = run_dir / "score_gen_pooled.svg"
    plot(gens, scores, out_path, f"{run_dir.name} — pool {args.n_pool}, proportion {args.proportion}")
    print(out_path)


if __name__ == "__main__":
    main()

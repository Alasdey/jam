"""
Derive per-population ratings from a tournament's stored payoff blocks.

Consumes only blocks/*.npz + tournament.json, so alternative rating schemes
(Elo, Nash averaging) can be added later as sibling functions over the same
artifacts. Self-play blocks are excluded from ratings by design.
Min/max statistics range over individual programs' mean payoffs and win rates.
"""

import json
import time
from pathlib import Path
import argparse

import numpy as np

from store.tournament import load_tournament


def load_block(tournament_dir: str, row_pop: str, col_pop: str, zero_sum: bool) -> np.ndarray:
    """Payoff oriented rows=row_pop, cols=col_pop, deriving -A.T when zero-sum."""
    blocks_dir = Path(tournament_dir) / "blocks"
    direct = blocks_dir / f"{row_pop}__vs__{col_pop}.npz"
    if direct.exists():
        with np.load(direct) as z:
            return z["payoff"]
    if zero_sum:
        reverse = blocks_dir / f"{col_pop}__vs__{row_pop}.npz"
        if reverse.exists():
            with np.load(reverse) as z:
                return -z["payoff"].T
    raise FileNotFoundError(
        f"No block for {row_pop} vs {col_pop} in {blocks_dir} (zero_sum={zero_sum})"
    )


def compute_ratings(tournament_dir: str) -> dict:
    doc = load_tournament(tournament_dir)
    members = doc["members"]

    populations = {}
    for pop in members:
        per_opponent = {}
        individual_sums = None
        individual_wins = None
        n_opponents = 0
        for opponent in members:
            if opponent == pop:
                continue
            block = load_block(tournament_dir, pop, opponent, doc["zero_sum"])
            row_sums = block.sum(axis=1, dtype=np.float64)
            row_wins = (block > 0).sum(axis=1)
            per_opponent[opponent] = _summarize(row_sums, row_wins, block.shape[1])
            if individual_sums is None:
                individual_sums = row_sums
                individual_wins = row_wins
            else:
                individual_sums += row_sums
                individual_wins += row_wins
            n_opponents += block.shape[1]
        if individual_sums is None:
            individual_sums = np.empty(0)
            individual_wins = np.empty(0)
        populations[pop] = {
            **_summarize(individual_sums, individual_wins, n_opponents),
            "per_opponent": per_opponent,
        }

    ranked = dict(
        sorted(populations.items(), key=lambda kv: kv[1]["mean_payoff"], reverse=True)
    )
    return {
        "format_version": 1,
        "tournament_id": doc["tournament_id"],
        "reward": doc["reward"],
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "populations": ranked,
    }


def _summarize(row_sums: np.ndarray, row_wins: np.ndarray, n_opponents: int) -> dict:
    """Summarize individuals against a common opponent pool, weighted by matchups."""
    n_matchups = len(row_sums) * n_opponents
    if n_matchups:
        payoffs = row_sums / n_opponents
        win_rates = row_wins / n_opponents
        return {
            "mean_payoff": round(float(row_sums.sum()) / n_matchups, 4),
            "min_payoff": round(float(payoffs.min()), 4),
            "max_payoff": round(float(payoffs.max()), 4),
            "win_rate": round(float(row_wins.sum()) / n_matchups, 4),
            "min_win_rate": round(float(win_rates.min()), 4),
            "max_win_rate": round(float(win_rates.max()), 4),
            "n_matchups": n_matchups,
        }
    return {
        "mean_payoff": 0.0,
        "min_payoff": 0.0,
        "max_payoff": 0.0,
        "win_rate": 0.0,
        "min_win_rate": 0.0,
        "max_win_rate": 0.0,
        "n_matchups": 0,
    }


def write_ratings(tournament_dir: str) -> dict:
    ratings = compute_ratings(tournament_dir)
    with open(Path(tournament_dir) / "ratings.json", "w") as f:
        json.dump(ratings, f, indent=2)
    return ratings

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tournament_dir", help="Tournament directory run dir, e.g. outputs/store/tournaments/<folder>")
    args = parser.parse_args()

    write_ratings(args.tournament_dir)


if __name__ == "__main__":
    main()

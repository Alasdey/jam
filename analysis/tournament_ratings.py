"""
Derive per-population ratings from a tournament's stored payoff blocks.

Consumes only blocks/*.npz + tournament.json, so alternative rating schemes
(Elo, Nash averaging) can be added later as sibling functions over the same
artifacts. Self-play blocks are excluded from ratings by design.
"""

import json
import time
from pathlib import Path

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
        total_sum = 0.0
        total_wins = 0
        total_matchups = 0
        for opponent in members:
            if opponent == pop:
                continue
            block = load_block(tournament_dir, pop, opponent, doc["zero_sum"])
            per_opponent[opponent] = {
                "mean_payoff": round(float(block.mean()), 4),
                "win_rate": round(float((block > 0).mean()), 4),
                "n_matchups": int(block.size),
            }
            total_sum += float(block.sum())
            total_wins += int((block > 0).sum())
            total_matchups += block.size
        populations[pop] = {
            "mean_payoff": round(total_sum / total_matchups, 4) if total_matchups else 0.0,
            "win_rate": round(total_wins / total_matchups, 4) if total_matchups else 0.0,
            "n_matchups": total_matchups,
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


def write_ratings(tournament_dir: str) -> dict:
    ratings = compute_ratings(tournament_dir)
    with open(Path(tournament_dir) / "ratings.json", "w") as f:
        json.dump(ratings, f, indent=2)
    return ratings

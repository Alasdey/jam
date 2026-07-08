"""Read side of the run-directory format written by ExperimentLogger."""

import json
import pickle
from pathlib import Path
from typing import Optional


def last_gen(run_dir: str) -> int:
    files = sorted(Path(run_dir).glob("populations/survivors_*.json"))
    if not files:
        raise FileNotFoundError(f"No survivors_*.json files in {run_dir}/populations")
    return int(files[-1].stem.rsplit("_", 1)[-1])


def load_survivors(run_dir: str, gen: int) -> dict:
    with open(Path(run_dir) / "populations" / f"survivors_{gen:06d}.json") as f:
        return json.load(f)


def load_births(run_dir: str, up_to_gen: int) -> dict[int, dict]:
    """id -> birth record, over all births files for generations <= up_to_gen."""
    births: dict[int, dict] = {}
    for path in sorted(Path(run_dir).glob("populations/births_*.jsonl")):
        gen = int(path.stem.rsplit("_", 1)[-1])
        if gen > up_to_gen:
            break
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    births[rec["id"]] = rec
    return births


def reconstruct_population(run_dir: str, gen: Optional[int] = None) -> list[dict]:
    """
    Individual records ({"id","genome","method","parents","born_gen"}) for the
    population surviving generation `gen` (latest if None), in survivor order.
    A survivor without a birth record raises KeyError — that means the run dir
    is inconsistent, not that a default should be invented.
    """
    if gen is None:
        gen = last_gen(run_dir)
    survivors = load_survivors(run_dir, gen)
    births = load_births(run_dir, up_to_gen=gen)
    return [births[i] for i in survivors["ids"]]


def load_checkpoint(run_dir: str) -> dict:
    with open(Path(run_dir) / "checkpoint.pkl", "rb") as f:
        return pickle.load(f)

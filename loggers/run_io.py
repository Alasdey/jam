"""Read side of the run-directory format written by ExperimentLogger."""

import json
import pickle
import re
from pathlib import Path
from typing import Iterable, Optional

_ID = re.compile(rb'\{"id":\s*(\d+)')


def last_gen(run_dir: str) -> int:
    files = sorted(Path(run_dir).glob("populations/survivors_*.json"))
    if not files:
        raise FileNotFoundError(f"No survivors_*.json files in {run_dir}/populations")
    return int(files[-1].stem.rsplit("_", 1)[-1])


def load_survivors(run_dir: str, gen: int) -> dict:
    with open(Path(run_dir) / "populations" / f"survivors_{gen:06d}.json") as f:
        return json.load(f)


def _births_files(run_dir: str, up_to_gen: int) -> list[Path]:
    files = []
    for path in sorted(Path(run_dir).glob("populations/births_*.jsonl")):
        if int(path.stem.rsplit("_", 1)[-1]) > up_to_gen:
            break
        files.append(path)
    return files


def load_births(run_dir: str, up_to_gen: int) -> dict[int, dict]:
    """
    id -> birth record, over all births files for generations <= up_to_gen.
    Holds every genome of the run in memory (gigabytes for long runs): for
    analyses of the whole history. To rebuild one population, use
    reconstruct_population, which reads only what it needs.
    """
    births: dict[int, dict] = {}
    for path in _births_files(run_dir, up_to_gen):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    births[rec["id"]] = rec
    return births


def find_births(run_dir: str, ids: Iterable[int], up_to_gen: int) -> dict[int, dict]:
    """
    Birth records of `ids`, reading births files from generation up_to_gen
    backwards and parsing only the lines of wanted ids, until all are found.
    """
    wanted = set(ids)
    found: dict[int, dict] = {}
    for path in reversed(_births_files(run_dir, up_to_gen)):
        with open(path, "rb") as f:
            for line in f:
                m = _ID.match(line)
                if m and int(m.group(1)) in wanted:
                    rec = json.loads(line)
                    if rec["id"] not in found:  # the latest record of an id wins, as in load_births
                        found[rec["id"]] = rec
        if len(found) == len(wanted):
            break
    return found


def reconstruct_population(run_dir: str, gen: Optional[int] = None) -> list[dict]:
    """
    Individual records ({"id","genome","method","parents","born_gen"}) for the
    population surviving generation `gen` (latest if None), in survivor order.
    A survivor without a birth record raises KeyError — that means the run dir
    is inconsistent, not that a default should be invented.
    """
    if gen is None:
        gen = last_gen(run_dir)
    ids = load_survivors(run_dir, gen)["ids"]
    births = find_births(run_dir, ids, up_to_gen=gen)
    return [births[i] for i in ids]


def max_birth_id(run_dir: str, up_to_gen: int) -> int:
    """The largest id in births files up to up_to_gen (-1 if none): ids only grow."""
    for path in reversed(_births_files(run_dir, up_to_gen)):
        best = -1
        with open(path, "rb") as f:
            for line in f:
                m = _ID.match(line)
                if m:
                    best = max(best, int(m.group(1)))
        if best >= 0:
            return best
    return -1


def load_checkpoint(run_dir: str) -> dict:
    with open(Path(run_dir) / "checkpoint.pkl", "rb") as f:
        return pickle.load(f)

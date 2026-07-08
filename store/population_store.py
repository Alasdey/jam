"""
Published sample populations, one directory per population:

  <store_dir>/<pop_id>/
    manifest.json       — provenance + full run config + compat/method keys
    individuals.jsonl   — one Individual record per line

pop_id = <label>__<YYYYMMDD_HHMMSS_us>__<compat_key[:8]>. Populations with
equal compat_key can play each other in a tournament; method_key groups
same-methodology samples published from different seeds.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import compat_key, method_key
from core.types import Program
from loggers.run_io import last_gen, reconstruct_population


def publish(store_dir: str, label: str, run_dir: str, gen: Optional[int] = None) -> str:
    """
    Copy the population surviving generation `gen` of run_dir (latest if None)
    into the store. Returns the new pop_id.
    """
    with open(Path(run_dir) / "config.json") as f:
        run_cfg = json.load(f)
    with open(Path(run_dir) / "meta.json") as f:
        meta = json.load(f)
    if gen is None:
        gen = last_gen(run_dir)
    records = reconstruct_population(run_dir, gen)

    ckey = compat_key(run_cfg["experiment"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pop_id = f"{label}__{stamp}__{ckey[:8]}"
    pop_dir = Path(store_dir) / pop_id
    pop_dir.mkdir(parents=True)  # an existing pop_dir is an error, not a target

    manifest = {
        "format_version": 1,
        "pop_id": pop_id,
        "label": label,
        "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_run": str(run_dir),
        "source_gen": gen,
        "n_individuals": len(records),
        "compat_key": ckey,
        "method_key": method_key(run_cfg),
        "seed": run_cfg["seed"],
        "git_hash": meta["git_hash"],
        "run_config": run_cfg,
    }
    with open(pop_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(pop_dir / "individuals.jsonl", "w") as f:
        for record in records:
            json.dump(record, f)
            f.write("\n")
    return pop_id


def load(store_dir: str, pop_id: str) -> tuple[dict, list[dict]]:
    """Returns (manifest, individual records)."""
    pop_dir = Path(store_dir) / pop_id
    with open(pop_dir / "manifest.json") as f:
        manifest = json.load(f)
    individuals = []
    with open(pop_dir / "individuals.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                individuals.append(json.loads(line))
    return manifest, individuals


def list_populations(store_dir: str) -> list[dict]:
    """Manifests of every published population (empty store -> empty list)."""
    store = Path(store_dir)
    if not store.exists():
        return []
    manifests = []
    for manifest_path in sorted(store.glob("*/manifest.json")):
        with open(manifest_path) as f:
            manifests.append(json.load(f))
    return manifests


def load_seed_genomes(store_dir: str, pop_ids: list[str]) -> tuple[list[Program], list[str]]:
    """
    Genomes for composite-run seeding, plus a parallel provenance list
    ("<pop_id>/<original_id>"). Seeded individuals receive local ids in this
    exact order starting at 0, so provenance[i] describes local id i.
    """
    genomes: list[Program] = []
    provenance: list[str] = []
    for pop_id in pop_ids:
        _, individuals = load(store_dir, pop_id)
        for record in individuals:
            genomes.append(record["genome"])
            provenance.append(f"{pop_id}/{record['id']}")
    return genomes, provenance

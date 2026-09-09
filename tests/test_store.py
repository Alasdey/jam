import json
from dataclasses import asdict
from pathlib import Path

import pytest

from config import (
    ExperimentConfig,
    RunConfig,
    SkimStepConfig,
    TreemoConfig,
)
from core.config_keys import compat_key, method_key
from core.loop import run
from loggers.run_io import reconstruct_population
from store.population_store import (
    list_populations,
    load,
    load_seed_genomes,
    publish,
)


def _exp() -> ExperimentConfig:
    return ExperimentConfig(
        interpreter="treemo_py",
        reward="placeholder",
        treemo=TreemoConfig(max_step=10, tree_size=15),
    )


def _cfg(out_dir, seed=123, **kw) -> RunConfig:
    return RunConfig(
        seed=seed,
        n_random=5,
        n_offspring=5,
        n_iter=3,
        selection=[SkimStepConfig(n_rounds=1, n_accepted=None)],
        out_dir=str(out_dir),
        experiment=_exp(),
        **kw,
    )


# ── keys ───────────────────────────────────────────────────────────────────


def test_compat_key_covers_execution_semantics_only():
    base = asdict(_exp())
    same = asdict(_exp())
    same["reward"] = "quine_pressure"
    same["genetics"]["mutation_rate"] = 0.9
    same["code"]["code_length"] = 7
    assert compat_key(base) == compat_key(same)

    changed = asdict(_exp())
    changed["treemo"]["max_step"] = 99
    assert compat_key(base) != compat_key(changed)

    other_interp = asdict(_exp())
    other_interp["interpreter"] = "treemo"
    assert compat_key(base) != compat_key(other_interp)


def test_method_key_ignores_run_identity_but_not_methodology(tmp_path):
    a = asdict(_cfg(tmp_path / "a", seed=1))
    b = asdict(_cfg(tmp_path / "b", seed=2))
    assert method_key(a) == method_key(b)

    other = _cfg(tmp_path / "c", seed=1)
    other.n_random = 9
    assert method_key(a) != method_key(asdict(other))


# ── publish / load ─────────────────────────────────────────────────────────


@pytest.fixture
def published(tmp_path):
    run_dir = tmp_path / "run"
    store_dir = tmp_path / "store"
    run(_cfg(run_dir))
    pop_id = publish(str(store_dir), "testpop", str(run_dir))
    return run_dir, store_dir, pop_id


def test_publish_load_round_trip(published):
    run_dir, store_dir, pop_id = published
    manifest, individuals = load(str(store_dir), pop_id)
    assert individuals == reconstruct_population(str(run_dir))
    assert manifest["pop_id"] == pop_id
    assert manifest["label"] == "testpop"
    assert manifest["source_gen"] == 2
    assert manifest["n_individuals"] == len(individuals)
    assert manifest["compat_key"] == compat_key(manifest["run_config"]["experiment"])
    assert manifest["seed"] == 123
    assert pop_id.endswith(manifest["compat_key"][:8])


def test_list_populations(published, tmp_path):
    _, store_dir, pop_id = published
    manifests = list_populations(str(store_dir))
    assert [m["pop_id"] for m in manifests] == [pop_id]
    assert list_populations(str(tmp_path / "nonexistent")) == []


def test_publish_specific_generation(published):
    run_dir, store_dir, _ = published
    pop_id = publish(str(store_dir), "gen0", str(run_dir), gen=0)
    manifest, individuals = load(str(store_dir), pop_id)
    assert manifest["source_gen"] == 0
    assert individuals == reconstruct_population(str(run_dir), 0)


# ── composite seeding ──────────────────────────────────────────────────────


def test_seed_populations_inject_store_individuals(published, tmp_path):
    _, store_dir, pop_id = published
    _, source_individuals = load(str(store_dir), pop_id)

    cfg = _cfg(tmp_path / "composite", seed=7)
    cfg.seed_populations = [pop_id]
    cfg.store_dir = str(store_dir)
    run(cfg)

    births0 = []
    with open(tmp_path / "composite" / "populations" / "births_000000.jsonl") as f:
        for line in f:
            births0.append(json.loads(line))
    seeded = [b for b in births0 if b["method"] == "seeded"]
    # seeded individuals get the first local ids, in store order (survivors of gen 0 only)
    genomes_by_id = {rec["id"]: rec["genome"] for rec in seeded}
    for local_id, source in enumerate(source_individuals):
        if local_id in genomes_by_id:
            assert genomes_by_id[local_id] == source["genome"]

    with open(tmp_path / "composite" / "meta.json") as f:
        meta = json.load(f)
    assert meta["seeded_provenance"] == [
        f"{pop_id}/{rec['id']}" for rec in source_individuals
    ]


def test_auto_publish_at_end_of_run(tmp_path):
    cfg = _cfg(tmp_path / "run", seed=11)
    cfg.publish_label = "auto"
    cfg.store_dir = str(tmp_path / "store")
    run(cfg)
    manifests = list_populations(str(tmp_path / "store"))
    assert len(manifests) == 1
    assert manifests[0]["label"] == "auto"
    _, individuals = load(str(tmp_path / "store"), manifests[0]["pop_id"])
    assert individuals == reconstruct_population(str(tmp_path / "run"))

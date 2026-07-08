import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from analysis.tournament_ratings import compute_ratings, load_block, write_ratings
from config import ExperimentConfig, RunConfig, SkimStepConfig, TreemoConfig
from core.loop import run
from interpreters import build_interpreter
from rewards.base import REWARDS
from store.population_store import load, publish
from store.tournament import add_populations, create_tournament, load_tournament

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _exp(max_step=10) -> ExperimentConfig:
    return ExperimentConfig(
        interpreter="treemo_py",
        reward="placeholder",
        treemo=TreemoConfig(max_step=max_step, tree_size=15),
    )


def _publish_run(tmp_path, label, seed, store_dir, max_step=10) -> str:
    cfg = RunConfig(
        seed=seed,
        n_random=4,
        n_offspring=4,
        n_iter=3,
        selection=[SkimStepConfig(n_rounds=1, n_accepted=None)],
        out_dir=str(tmp_path / f"run_{label}"),
        experiment=_exp(max_step),
    )
    run(cfg)
    return publish(str(store_dir), label, cfg.out_dir)


@pytest.fixture
def arena(tmp_path):
    store = tmp_path / "store"
    pop_a = _publish_run(tmp_path, "alpha", 1, store)
    pop_b = _publish_run(tmp_path, "beta", 2, store)
    tdir = tmp_path / "tournament"
    create_tournament(str(tdir), "placeholder", pop_a, str(store))
    add_populations(str(tdir), [pop_b])
    return store, tdir, pop_a, pop_b


def test_create_and_add_store_one_canonical_block(arena, tmp_path):
    store, tdir, pop_a, pop_b = arena
    doc = load_tournament(str(tdir))
    assert doc["members"] == [pop_a, pop_b]
    assert doc["zero_sum"] is True

    row_pop, col_pop = sorted([pop_a, pop_b])
    blocks = sorted(p.name for p in (tdir / "blocks").glob("*.npz"))
    assert blocks == [f"{row_pop}__vs__{col_pop}.npz"]

    with np.load(tdir / "blocks" / blocks[0]) as z:
        payoff, row_ids, col_ids = z["payoff"], z["row_ids"], z["col_ids"]
    _, row_individuals = load(str(store), row_pop)
    _, col_individuals = load(str(store), col_pop)
    assert row_ids.tolist() == [r["id"] for r in row_individuals]
    assert col_ids.tolist() == [r["id"] for r in col_individuals]
    assert payoff.shape == (len(row_individuals), len(col_individuals))

    with open(tdir / "blocks" / f"{row_pop}__vs__{col_pop}.json") as f:
        sidecar = json.load(f)
    assert sidecar["row_pop"] == row_pop
    assert sidecar["reward"] == "placeholder"


def test_block_matches_direct_evaluation_and_reverse_is_negated(arena):
    store, tdir, pop_a, pop_b = arena
    _, a_individuals = load(str(store), pop_a)
    _, b_individuals = load(str(store), pop_b)

    interp = build_interpreter(_exp())
    fn = REWARDS["placeholder"].fn
    expected = np.array([
        [fn(interp, ra["genome"], rb["genome"]) for rb in b_individuals]
        for ra in a_individuals
    ])
    assert np.array_equal(load_block(str(tdir), pop_a, pop_b, True), expected)
    assert np.array_equal(load_block(str(tdir), pop_b, pop_a, True), -expected.T)


def test_add_is_incremental(arena, tmp_path):
    store, tdir, pop_a, pop_b = arena
    before = {p.name: p.read_bytes() for p in (tdir / "blocks").glob("*.npz")}

    pop_c = _publish_run(tmp_path, "gamma", 3, store)
    add_populations(str(tdir), [pop_c])

    after = {p.name: p.read_bytes() for p in (tdir / "blocks").glob("*.npz")}
    for name, data in before.items():
        assert after[name] == data  # existing blocks untouched
    assert len(after) == len(before) + 2  # c-vs-a and c-vs-b only
    assert load_tournament(str(tdir))["members"] == [pop_a, pop_b, pop_c]


def test_add_rejects_incompatible_population(arena, tmp_path):
    store, tdir, _, _ = arena
    pop_other = _publish_run(tmp_path, "other", 4, store, max_step=99)
    with pytest.raises(ValueError, match="compat_key"):
        add_populations(str(tdir), [pop_other])


def test_add_rejects_duplicate_member(arena):
    _, tdir, pop_a, _ = arena
    with pytest.raises(ValueError, match="already"):
        add_populations(str(tdir), [pop_a])


def test_self_play_blocks_are_optional(tmp_path):
    store = tmp_path / "store"
    pop_a = _publish_run(tmp_path, "alpha", 1, store)
    tdir = tmp_path / "tournament"
    create_tournament(str(tdir), "placeholder", pop_a, str(store), include_self=True)
    assert (tdir / "blocks" / f"{pop_a}__vs__{pop_a}.npz").exists()


def test_ratings_on_hand_built_blocks(tmp_path):
    tdir = tmp_path / "t"
    (tdir / "blocks").mkdir(parents=True)
    doc = {
        "format_version": 1, "tournament_id": "t", "reward": "placeholder",
        "zero_sum": True, "compat_key": "x", "store_dir": "unused",
        "experiment_config": {}, "members": ["a", "b"],
    }
    with open(tdir / "tournament.json", "w") as f:
        json.dump(doc, f)
    np.savez(
        tdir / "blocks" / "a__vs__b.npz",
        payoff=np.ones((3, 2), dtype=np.int32),  # a always beats b
        row_ids=np.arange(3), col_ids=np.arange(2),
    )
    ratings = compute_ratings(str(tdir))
    pops = ratings["populations"]
    assert list(pops) == ["a", "b"]  # ranked by mean payoff
    assert pops["a"]["mean_payoff"] == 1.0
    assert pops["a"]["win_rate"] == 1.0
    assert pops["b"]["mean_payoff"] == -1.0
    assert pops["b"]["win_rate"] == 0.0
    assert pops["a"]["per_opponent"]["b"]["n_matchups"] == 6


def test_ratings_end_to_end_antisymmetric(arena):
    _, tdir, pop_a, pop_b = arena
    ratings = write_ratings(str(tdir))
    assert (tdir / "ratings.json").exists()
    pops = ratings["populations"]
    ab = pops[pop_a]["per_opponent"][pop_b]["mean_payoff"]
    ba = pops[pop_b]["per_opponent"][pop_a]["mean_payoff"]
    assert ab == pytest.approx(-ba)
    assert pops[pop_a]["n_matchups"] == pops[pop_b]["n_matchups"]


def test_cli_round_trip(tmp_path):
    store = tmp_path / "store"
    pop_a = _publish_run(tmp_path, "alpha", 1, store)

    run_b = RunConfig(
        seed=2, n_random=4, n_offspring=4, n_iter=3,
        selection=[SkimStepConfig(n_rounds=1, n_accepted=None)],
        out_dir=str(tmp_path / "run_cli"), experiment=_exp(),
    )
    run(run_b)

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", *args],
            cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=True,
        ).stdout.strip()

    pop_b = cli("store.publish", run_b.out_dir, "--label", "cli-beta", "--store", str(store))
    tdir = tmp_path / "tournament"
    cli("store.tournament", "create", "--dir", str(tdir),
        "--reward", "placeholder", "--from-pop", pop_a, "--store", str(store))
    cli("store.tournament", "add", "--dir", str(tdir), pop_b)
    table = cli("store.tournament", "ratings", "--dir", str(tdir))
    assert pop_a in table and pop_b in table
    assert (tdir / "ratings.json").exists()

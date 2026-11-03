import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from config import CapStepConfig, ExperimentConfig, PayoffConfig, RunConfig, TreemoConfig
from core.config_keys import method_key
from core.loop import run
from loggers.run_io import last_gen, load_checkpoint, reconstruct_population
from rewards.base import build_reward
from rewards.payoff import PayoffEngine


def _cfg(path, **overrides):
    values = dict(
        out_dir=str(path), resume_from=None, publish_label=None,
        seed=123, n_random=3, n_offspring=2, n_iter=5, payoff_every=2,
        selection=[CapStepConfig(max_pop=8)],
        experiment=ExperimentConfig(
            interpreter="treemo", reward="placeholder",
            treemo=TreemoConfig(max_step=4, tree_size=8),
            payoff=PayoffConfig(),
        ),
    )
    values.update(overrides)
    return RunConfig(**values)


def _files(path):
    return {str(p.relative_to(path)): p.read_bytes()
            for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("gen", [0, 2])
def test_snapshot_restart_keeps_population_and_complete_backup(tmp_path, monkeypatch, gen):
    path = tmp_path / "run"
    run(_cfg(path))
    records = reconstruct_population(str(path), gen)
    previous = load_checkpoint(str(path))
    before = _files(path)
    with np.load(path / "payoffs" / f"payoff_{gen:06d}.npz") as data:
        expected = data["payoff"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical resume must not recompute a valid snapshot")

    monkeypatch.setattr(PayoffEngine, "square", forbidden)
    run(_cfg(path, resume_from=str(path), resume_gen=gen, n_iter=0))
    assert last_gen(str(path)) == gen
    assert reconstruct_population(str(path)) == records
    ck = load_checkpoint(str(path))
    assert ck["gen"] == gen + 1
    assert ck["next_id"] == previous["next_id"]
    np.testing.assert_array_equal(ck["payoff"], expected)
    backups = list(tmp_path.glob(".run.before_gen_*"))
    assert len(backups) == 1
    assert _files(backups[0]) == before
    for name in ("config.json", "meta.json", f"populations/survivors_{gen:06d}.json"):
        assert _files(path)[name] == before[name]


def test_missing_payoff_recomputes_and_normal_resume_follows_new_history(tmp_path, monkeypatch):
    path = tmp_path / "run"
    run(_cfg(path))
    old_next_id = load_checkpoint(str(path))["next_id"]
    original = PayoffEngine.square
    calls = []

    def record(self, pop, known=None, n_known=0):
        calls.append((len(pop), n_known))
        return original(self, pop, known, n_known)

    monkeypatch.setattr(PayoffEngine, "square", record)
    records = reconstruct_population(str(path), 1)
    run(_cfg(path, resume_from=str(path), resume_gen=1, n_iter=2))
    assert calls[0] == (len(records), 0)  # the whole matrix, recomputed
    assert last_gen(str(path)) == 3
    assert [json.loads(line)["gen"] for line in (path / "metrics.jsonl").read_text().splitlines()] == [0, 1, 2, 3]
    assert not (path / "populations/births_000004.jsonl").exists()
    assert not (path / "payoffs/payoff_000004.npz").exists()
    for gen in (2, 3):
        p = path / "populations" / f"births_{gen:06d}.jsonl"
        if p.exists():
            assert all(json.loads(line)["id"] >= old_next_id for line in p.read_text().splitlines())
    run(_cfg(path, resume_from=str(path), n_iter=1))
    assert last_gen(str(path)) == 4
    assert load_checkpoint(str(path))["gen"] == 5


def test_restart_without_checkpoint_or_snapshot(tmp_path):
    path = tmp_path / "run"
    run(_cfg(path))
    (path / "checkpoint.pkl").unlink()
    records = reconstruct_population(str(path), 1)
    cfg = _cfg(path, resume_from=str(path), resume_gen=1, n_iter=0)
    genomes = [r["genome"] for r in records]
    with PayoffEngine(cfg.experiment, build_reward(cfg.experiment)) as engine:
        expected = engine.matrix(genomes, genomes)
    run(cfg)
    ck = load_checkpoint(str(path))
    assert ck["gen"] == 2
    assert ck["next_id"] > max(r["id"] for r in records)
    np.testing.assert_array_equal(ck["payoff"], expected)


@pytest.mark.parametrize("value", [-1, 1.5, True, "2", 99])
def test_invalid_generation_leaves_run_untouched(tmp_path, value):
    path = tmp_path / "run"
    run(_cfg(path))
    before = _files(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        run(_cfg(path, resume_from=str(path), resume_gen=value, n_iter=0))
    assert _files(path) == before
    assert not list(tmp_path.glob(".run.before_gen_*"))


def test_generation_requires_source(tmp_path):
    path = tmp_path / "new"
    with pytest.raises(ValueError, match="requires resume_from"):
        run(_cfg(path, resume_gen=0))
    assert not path.exists()


def test_corrupt_snapshot_rejected_before_rewind(tmp_path):
    path = tmp_path / "run"
    run(_cfg(path))
    np.savez(path / "payoffs/payoff_000002.npz", payoff=np.zeros((1, 1)), ids=[-1])
    before = _files(path)
    with pytest.raises(ValueError, match="IDs"):
        run(_cfg(path, resume_from=str(path), resume_gen=2, n_iter=0))
    assert _files(path) == before
    assert not list(tmp_path.glob(".run.before_gen_*"))


def test_failed_recomputation_leaves_run_untouched(tmp_path, monkeypatch):
    path = tmp_path / "run"
    run(_cfg(path))
    before = _files(path)

    def fail(*args):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(PayoffEngine, "square", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        run(_cfg(path, resume_from=str(path), resume_gen=1, n_iter=0))
    assert _files(path) == before
    assert not list(tmp_path.glob(".run.before_gen_*"))


def test_rewind_io_failure_restores_previous_run(tmp_path, monkeypatch):
    import loggers.rewind as module

    path = tmp_path / "run"
    run(_cfg(path))
    before = _files(path)

    def fail(*args):
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(module, "write_checkpoint", fail)
    with pytest.raises(OSError, match="checkpoint write failed"):
        run(_cfg(path, resume_from=str(path), resume_gen=2, n_iter=0))
    assert _files(path) == before


def test_resume_generation_is_not_part_of_method_key(tmp_path):
    a = asdict(_cfg(tmp_path / "run"))
    b = dict(a, resume_gen=399)
    assert method_key(a) == method_key(b)
    del a["resume_gen"]  # Keys of pre-feature run configs remain unchanged.
    assert method_key(a) == method_key(b)

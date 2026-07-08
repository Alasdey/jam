import json
from pathlib import Path

import numpy as np
import pytest

from config import (
    PRESETS,
    CapStepConfig,
    ExperimentConfig,
    RunConfig,
    SkimStepConfig,
    TreemoConfig,
)
from core.loop import run
from loggers.run_io import last_gen, load_checkpoint, reconstruct_population

TIME_FIELDS = ("t", "create_s", "payoff_s", "select_s")


def _exp() -> ExperimentConfig:
    return ExperimentConfig(
        interpreter="treemo_py",
        reward="placeholder",
        treemo=TreemoConfig(max_step=10, tree_size=15),
    )


def _cfg(out_dir, seed=123, n_iter=5, **kw) -> RunConfig:
    return RunConfig(
        seed=seed,
        n_random=5,
        n_offspring=5,
        n_iter=n_iter,
        selection=[SkimStepConfig(n_rounds=2, n_accepted=None), CapStepConfig(max_pop=12)],
        out_dir=str(out_dir),
        experiment=_exp(),
        **kw,
    )


def _read_metrics(run_dir, drop_times=True):
    records = []
    with open(Path(run_dir) / "metrics.jsonl") as f:
        for line in f:
            record = json.loads(line)
            if drop_times:
                for field in TIME_FIELDS:
                    record.pop(field)
            records.append(record)
    return records


def _read_pop_files(run_dir):
    pops = Path(run_dir) / "populations"
    return {p.name: p.read_text() for p in sorted(pops.iterdir())}


def test_same_seed_runs_are_identical(tmp_path):
    run(_cfg(tmp_path / "a"))
    run(_cfg(tmp_path / "b"))
    assert _read_metrics(tmp_path / "a") == _read_metrics(tmp_path / "b")
    assert _read_pop_files(tmp_path / "a") == _read_pop_files(tmp_path / "b")


def test_different_seeds_diverge(tmp_path):
    run(_cfg(tmp_path / "a", seed=1))
    run(_cfg(tmp_path / "b", seed=2))
    assert _read_pop_files(tmp_path / "a") != _read_pop_files(tmp_path / "b")


def test_resolved_seed_is_persisted(tmp_path):
    run(_cfg(tmp_path / "r", seed=None, n_iter=1))
    with open(tmp_path / "r" / "config.json") as f:
        saved = json.load(f)
    assert isinstance(saved["seed"], int)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    run(_cfg(tmp_path / "full", n_iter=6))
    run(_cfg(tmp_path / "split", n_iter=3))
    run(_cfg(tmp_path / "split", n_iter=3, resume_from=str(tmp_path / "split")))

    assert last_gen(tmp_path / "split") == last_gen(tmp_path / "full") == 5
    assert reconstruct_population(str(tmp_path / "split")) == reconstruct_population(
        str(tmp_path / "full")
    )
    ck_full = load_checkpoint(str(tmp_path / "full"))
    ck_split = load_checkpoint(str(tmp_path / "split"))
    assert ck_full["next_id"] == ck_split["next_id"]
    assert np.array_equal(ck_full["payoff"], ck_split["payoff"])
    # the resumed generations produced identical metrics
    assert _read_metrics(tmp_path / "full")[3:] == _read_metrics(tmp_path / "split")[3:]


def test_resume_rejects_incompatible_interpreter_config(tmp_path):
    run(_cfg(tmp_path / "r", n_iter=2))
    resumed = _cfg(tmp_path / "r", n_iter=1, resume_from=str(tmp_path / "r"))
    resumed.experiment.treemo = TreemoConfig(max_step=99, tree_size=15)
    with pytest.raises(ValueError, match="compat_key"):
        run(resumed)


def test_population_reconstructable_at_every_generation(tmp_path):
    cfg = _cfg(tmp_path / "r")
    run(cfg)
    for metric in _read_metrics(tmp_path / "r"):
        records = reconstruct_population(str(tmp_path / "r"), metric["gen"])
        assert len(records) == metric["pop_size"]
        for record in records:
            assert set(record) == {"id", "genome", "method", "parents", "born_gen"}
    checkpoint = load_checkpoint(str(tmp_path / "r"))
    final = reconstruct_population(str(tmp_path / "r"))
    assert [record["id"] for record in final] == checkpoint["pop_ids"]


def test_payoff_matrices_persisted_periodically_and_at_end(tmp_path):
    cfg = _cfg(tmp_path / "p", n_iter=4)
    cfg.payoff_every = 2
    run(cfg)
    names = sorted(p.name for p in (tmp_path / "p" / "payoffs").iterdir())
    assert names == ["payoff_000000.npz", "payoff_000002.npz", "payoff_000003.npz"]

    with np.load(tmp_path / "p" / "payoffs" / "payoff_000003.npz") as z:
        payoff, ids = z["payoff"], z["ids"]
    with open(tmp_path / "p" / "populations" / "survivors_000003.json") as f:
        survivors = json.load(f)
    assert ids.tolist() == survivors["ids"]
    assert payoff.shape == (len(ids), len(ids))
    # placeholder reward is zero-sum: the self-play matrix is antisymmetric
    assert np.array_equal(payoff, -payoff.T)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_smoke(tmp_path, name):
    cfg = PRESETS[name]()
    cfg.experiment = _exp()
    cfg.seed = 5
    cfg.n_iter = 2
    cfg.n_init = min(cfg.n_init, 4)
    cfg.n_random = min(cfg.n_random, 4)
    cfg.n_offspring = min(cfg.n_offspring, 4)
    cfg.out_dir = str(tmp_path / name)
    run(cfg)
    assert last_gen(cfg.out_dir) == 1
    assert _read_metrics(cfg.out_dir)[-1]["pop_size"] > 0

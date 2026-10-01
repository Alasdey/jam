import json
import random
from dataclasses import asdict

import numpy as np

import pytest

import analysis.score_gen_pooled as pooled
from analysis.score_gen_pooled import sample_unique_pool
from config import ExperimentConfig


def test_duplicate_births_do_not_change_sampling_weight():
    births = {0: {"genome": [1, 0]}, 1: {"genome": [1, 1, 0, 0]}, 2: {"genome": []}}
    repeated = {**births, **{i: {"genome": [1, 0]} for i in range(3, 100)}}
    for seed in range(20):
        assert sample_unique_pool(births, 2, random.Random(seed)) == sample_unique_pool(
            repeated, 2, random.Random(seed)
        )
    assert len({tuple(g) for g in sample_unique_pool(repeated, 3, random.Random(0))}) == 3


@pytest.mark.parametrize("count", [0, -1, 2])
def test_pool_size_is_checked_against_distinct_genomes(count):
    births = {0: {"genome": [1, 0]}, 1: {"genome": [1, 0]}}
    with pytest.raises(ValueError, match="distinct persisted programs"):
        sample_unique_pool(births, count, random.Random(0))


def test_scores_are_reused_for_identical_codes_during_one_run(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"experiment": asdict(ExperimentConfig())}))
    births = {1: {"genome": [1]}, 2: {"genome": [1]}, 3: {"genome": [2]}}
    evaluated = []

    class FakeEngine:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def matrix(self, codes, pool):
            evaluated.extend(codes)
            return np.array([[sum(code)] for code in codes])

    monkeypatch.setattr(pooled, "PayoffEngine", FakeEngine)
    monkeypatch.setattr(pooled, "last_gen", lambda run_dir: 1)
    monkeypatch.setattr(pooled, "load_births", lambda run_dir, up_to_gen: births)
    monkeypatch.setattr(pooled, "sample_unique_pool", lambda births, n_pool, rng: [[9]])
    monkeypatch.setattr(pooled, "generations", lambda run_dir: [0, 1])
    monkeypatch.setattr(pooled, "load_survivors", lambda run_dir, gen: {"ids": [1, 2] if gen == 0 else [2, 3]})

    gens, scores = pooled.score_gen_pooled(tmp_path, 1, 1.0, 1, 0)

    assert evaluated == [[1], [2]]
    assert gens == [0, 0, 1, 1]
    assert scores[:2] == [1.0, 1.0]
    assert sorted(scores[2:]) == [1.0, 2.0]

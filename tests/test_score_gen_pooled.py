import random

import pytest

from analysis.score_gen_pooled import sample_unique_pool


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

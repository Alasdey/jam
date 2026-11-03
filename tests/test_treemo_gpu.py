import random

import numpy as np
import pytest

from core.matchups import lcs_length
from creation.treemo import gen_tree
from interpreters.treemo_c.treemo import TreemoInterpreter as CTreemo
from interpreters.treemo_gpu.treemo import TreemoInterpreter, extract_rules
from interpreters.treemo_py.treemo import tree_to_rules


def _dp_lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def _similarity(output, reference):
    return _dp_lcs(output, reference) / len(reference) if output and reference else 0.0

def _programs(n, seed, max_nodes=60):
    rng = random.Random(seed)
    random.seed(seed)  # gen_tree draws from the global RNG
    progs = [gen_tree(rng.randint(1, max_nodes)) for _ in range(n)]
    # forests: the implicit root then has several children
    progs += [sum((gen_tree(rng.randint(1, 10)) for _ in range(rng.randint(2, 4))), [])
              for _ in range(n // 4)]
    return progs + [[], [1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0, 1, 0]]


def _pairs(n, seed):
    progs = _programs(n, seed)
    rng = random.Random(seed + 1)
    pairs = [(rng.choice(progs), rng.choice(progs)) for _ in range(4 * n)]
    return [a for a, _ in pairs], [b for _, b in pairs]


def test_extract_rules_matches_reference():
    progs = _programs(400, seed=11)
    for code, table in zip(progs, extract_rules([bytes(p) for p in progs])):
        rules, ident = table
        got = [(code[ps:ps + pl], code[rs:rs + rl]) for ps, pl, rs, rl in rules.tolist()]
        expected = [(list(m), list(r)) for m, r in (tree_to_rules(code) if code else [])]
        assert got == expected
        assert ident.tolist() == [int(m == r) for m, r in expected]


def test_extract_rules_rejects_unbalanced_and_nonbinary():
    tables = extract_rules([b"\x01", b"\x00\x01", b"\x01\x00\x00\x01", b"\x01\x02\x00\x00"])
    assert tables == [None, None, None, None]


def test_host_lcs_matches_dynamic_programming():
    rng = random.Random(5)
    for _ in range(300):
        a = bytes(rng.choice([0, 1, 2]) for _ in range(rng.randint(0, 50)))
        b = bytes(rng.choice([0, 1, 2]) for _ in range(rng.randint(0, 50)))
        assert lcs_length(a, b) == _dp_lcs(list(a), list(b))


@pytest.mark.cuda
@pytest.mark.parametrize("max_step", [0, 7, 50])
@pytest.mark.parametrize("pass_mode", [False, True])
@pytest.mark.parametrize("first_mode", [False, True])
def test_outputs_and_lengths_match_c(max_step, pass_mode, first_mode):
    codes, inputs = _pairs(300, seed=max_step + 2 * pass_mode + 4 * first_mode)
    c = CTreemo(max_step, pass_mode, first_mode)
    gpu = TreemoInterpreter(max_step, pass_mode, first_mode)
    expected = [c.run(a, b)[0] for a, b in zip(codes, inputs)]
    got = gpu.run_batch(codes, inputs)
    assert gpu.last_stats["cpu_runs"] == 0
    assert [out for out, _ in got] == expected
    assert all(code is original for (_, code), original in zip(got, codes))
    assert gpu.lengths(codes, inputs).tolist() == [len(out) for out in expected]


@pytest.mark.cuda
def test_self_similarities_match_reference():
    codes, inputs = _pairs(150, seed=21)
    c = CTreemo(50, False, False)
    gpu = TreemoInterpreter(50, False, False)
    expected = [_similarity(c.run(a, b)[0], a) for a, b in zip(codes, inputs)]
    assert gpu.self_similarities(codes, inputs).tolist() == expected


@pytest.mark.cuda
def test_states_that_grow_far_past_their_input():
    # One rule that doubles a leaf each step: output length = input + 2 * max_step.
    code = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0]  # rule (10, 1010)
    gpu = TreemoInterpreter(max_step=500, pass_mode=False, first_mode=False)
    c = CTreemo(500, False, False)
    inputs = [[1, 0], [1, 1, 0, 0] * 40, []]
    assert [out for out, _ in gpu.run_batch([code] * 3, inputs)] == [c.run(code, i)[0] for i in inputs]


@pytest.mark.cuda
def test_unsupported_tapes_fall_back_to_c():
    codes, inputs = _pairs(40, seed=31)
    codes = codes + [[1, 0, 0, 1], [1, 1, 0, 0]]  # an unbalanced program
    inputs = inputs + [[1, 1, 0, 0], [0, 1, 2, 3, 4, 5]]  # a non-binary input
    c = CTreemo(50, False, False)
    gpu = TreemoInterpreter(50, False, False)
    assert [out for out, _ in gpu.run_batch(codes, inputs)] == [c.run(a, b)[0] for a, b in zip(codes, inputs)]
    assert gpu.last_stats["cpu_runs"] == 2


@pytest.mark.cuda
def test_small_memory_budget_splits_launches():
    random.seed(41)
    progs = [gen_tree(random.randint(200, 400)) for _ in range(60)]
    codes = [p for p in progs for _ in progs]
    inputs = [q for _ in progs for q in progs]  # 3,600 jobs: several MB of state
    one = TreemoInterpreter(50, False, False)
    many = TreemoInterpreter(50, False, False, memory_mb=1)
    expected = one.self_similarities(codes, inputs)
    assert many.self_similarities(codes, inputs).tolist() == expected.tolist()
    assert many.last_stats["launches"] > one.last_stats["launches"]


@pytest.mark.cuda
def test_long_tapes_score_on_the_device_unless_both_are_long():
    random.seed(3)
    code = gen_tree(8300)  # 16,598 bits: beyond the device LCS limit of 16,384
    short, long = gen_tree(30), gen_tree(8300)
    gpu = TreemoInterpreter(0, False, False)  # no steps: the output is the input
    # The shorter tape is the bit-vector side, so a long program with a short
    # output stays on the device; two long tapes go to the host.
    for inp, host in ((short, 0), (long, 1)):
        expected = lcs_length(inp, code) / len(code)
        assert gpu.self_similarities([code], [inp]).tolist() == [expected]
        assert gpu.last_stats["host_measures"] == host


@pytest.mark.cuda
def test_empty_batch():
    gpu = TreemoInterpreter()
    assert gpu.run_batch([], []) == []
    assert gpu.lengths([], []).shape == (0,)

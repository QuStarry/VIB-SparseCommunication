import pytest

from experiments.rescore_repaired_plus import HELPER, OLD, NEW, repair_task32
from src.isolated_plus_evaluation import score_plus


def task32():
    return {"task_id": "HumanEval/32", "entry_point": "find_zero",
            "test": "def _poly(xs, x):\n    return xs[0] + xs[1] * x\n"
                    "def check(candidate):\n    inputs = [[[-2, 1]]]\n    results = [2]\n"
                    "    for inp, exp in zip(inputs, results):\n        " + OLD,
            "prompt": "public contract", "canonical_solution": "reference unchanged"}


def test_repair_changes_only_unique_assertion():
    source = task32()
    repaired = repair_task32(source)
    assert source["test"].endswith(OLD)
    assert repaired == {**source, "test": source["test"].replace(OLD, NEW) + "\n" + HELPER}


def test_repair_rejects_wrong_task_or_duplicate():
    with pytest.raises(ValueError):
        repair_task32({**task32(), "task_id": "HumanEval/0"})
    with pytest.raises(ValueError):
        repair_task32({**task32(), "test": OLD + "\n" + OLD})


def test_correct_root_passes():
    assert score_plus("def find_zero(xs): return 2.0", repair_task32(task32()))["passed"]


def test_large_negative_residual_does_not_pass():
    assert not score_plus("def find_zero(xs): return -100.0", repair_task32(task32()))["passed"]


def test_large_positive_residual_does_not_pass():
    assert not score_plus("def find_zero(xs): return 100.0", repair_task32(task32()))["passed"]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_root_fails(value):
    assert not score_plus(f"def find_zero(xs): return float('{value}')", repair_task32(task32()))["passed"]


def test_exact_neighbor_sign_change_accepts_unavoidable_float_residual():
    namespace = {}
    exec(HELPER, namespace)
    coefficients = [9450000, 9, -7, 3, 2, 6, 1, -4, -10, 6, 17, -1]
    assert namespace["_vib_local_root_check"](coefficients, 17.3124550475086)
    assert not namespace["_vib_local_root_check"](coefficients, 17.31)


def test_precision_guard_does_not_accept_constant_nonzero_polynomial():
    namespace = {}
    exec(HELPER, namespace)
    assert not namespace["_vib_local_root_check"]([0.001], 10.0)


def test_boolean_and_nonnumeric_root_rejected():
    namespace = {}
    exec(HELPER, namespace)
    for value in (True, "2.0", [2.0]):
        assert not namespace["_vib_local_root_check"]([-2, 1], value)

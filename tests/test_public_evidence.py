"""Public certificate checks are separate from benchmark reference evaluation."""
import random

from experiments.audit_public_evidence import (
    audit_case, execute_public_checks, public_spec, public_view,
)


def test_public_view_does_not_read_hidden_material():
    class GuardedTask(dict):
        def __getitem__(self, key):
            if key in ("test_cases", "reference", "entry_point"):
                raise AssertionError("Hidden material must not be accessed")
            return super().__getitem__(key)

    task = GuardedTask(task_id="HumanEval/93", problem="public", candidates=[])
    assert public_view(task) == dict(task)
    case = audit_case(task, [])
    assert "test_cases" not in case
    assert "reference" not in case


def test_public_example_rejects_wrong_vowel_case():
    checks = public_spec("HumanEval/93")["checks"]
    wrong = "def encode(message): return 'TgST'"
    right = "def encode(message): return 'TGST'"
    assert not execute_public_checks(wrong, checks)["checks"][0]["satisfied"]
    assert execute_public_checks(right, checks)["checks"][0]["satisfied"]


def test_nonnegative_counterexample_rejects_fixed_width_mask():
    checks = public_spec("HumanEval/116")["checks"]
    wrong = "def sort_array(arr): return sorted(arr, key=lambda x: (bin(x & 0xffffffff).count('1'), x))"
    right = "def sort_array(arr): return sorted(arr, key=lambda x: (bin(abs(x)).count('1'), x))"
    assert not execute_public_checks(wrong, checks)["checks"][0]["satisfied"]
    assert execute_public_checks(right, checks)["checks"][0]["satisfied"]


def test_sort_specification_conflict_is_explicit():
    certificate = public_spec("HumanEval/116")
    conflict = certificate["specification_conflicts"][0]
    assert conflict["literal_example"] != conflict["rule_derived"]
    assert "negative-number" in certificate["interpretation"]


def test_triple_counterexample_uses_direct_problem_enumeration():
    checks = public_spec("HumanEval/147")["checks"]
    brute_force = (
        "from itertools import combinations\n"
        "def get_max_triples(n):\n"
        "    values = [i*i-i+1 for i in range(1,n+1)]\n"
        "    return sum(sum(triple)%3 == 0 for triple in combinations(values,3))"
    )
    results = execute_public_checks(brute_force, checks)["checks"]
    assert [row["actual"] for row in results] == [1, 0]
    assert all(row["satisfied"] for row in results)


def test_math_certificate_validates_areas_and_minimum():
    certificate = public_spec("MATH/1791")
    assert certificate["verified_areas"] == ["18", "72"]
    assert certificate["derived_minimum"] == 16
    assert all(2*k*k + 12*k + 2 > 16 for k in range(2, 100))
    task = {"task_id": "MATH/1791", "problem": "public", "candidates": []}
    rows = [
        {"policy": "full", "repeat": 0, "request_id": "1", "output": "答案: 16"},
        {"policy": "first_round", "repeat": 0, "request_id": "2", "output": "答案: 20"},
    ]
    by_policy = {row["policy"]: row for row in audit_case(task, rows)["outputs"]}
    assert by_policy["full"]["public_evaluation"]["satisfied"]
    assert not by_policy["first_round"]["public_evaluation"]["satisfied"]


def test_child_execution_does_not_mutate_parent_random_state():
    state = random.getstate()
    try:
        result = execute_public_checks(
            "import random\nrandom.seed(999)\ndef answer(): return 1",
            [{"expression": "answer()", "expected": 1}],
        )
        assert result["checks"][0]["satisfied"]
        assert random.getstate() == state
    finally:
        random.setstate(state)


def test_execution_error_is_not_reported_as_public_counterexample():
    result = execute_public_checks("raise RuntimeError('bad')", [])
    assert result["status"] == "execution_error"
    assert "checks" not in result


def test_unknown_task_remains_unsupported():
    task = {"task_id": "new", "problem": "public", "candidates": []}
    report = audit_case(task, [])
    assert report["certificate"]["source"] == "unsupported"
    assert report["candidate_message_attribution"] == "not_intervened_not_proven"

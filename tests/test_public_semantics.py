from experiments.audit_public_evidence import execute_public_checks
from experiments.public_semantics import property_output, semantic_contract


POLYNOMIAL_PROBLEM = '''
import math
def poly(xs, x):
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])
def solve(xs):
    """find x such that poly(x) = 0.
    >>> round(solve([-6, 11, -6, 1]), 2)
    1.0
    """
'''

SORT_PROBLEM = '''
def order(values):
    """Sort non-negative integers by number of ones in binary representation
    in ascending order; ties use decimal value.
    >>> order([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]
    """
'''


def test_any_polynomial_root_is_accepted_not_only_example_value():
    contract = semantic_contract(POLYNOMIAL_PROBLEM)
    assert contract["mode"] == "polynomial_root_residual"
    for root in (1, 2, 3):
        code = property_output(f"def solve(xs): return {root}", contract["support_code"])
        assert execute_public_checks(code, contract["checks"])["checks"][0]["satisfied"]


def test_nonroot_is_rejected_even_if_rounded_to_displayed_value():
    contract = semantic_contract(POLYNOMIAL_PROBLEM)
    code = property_output("def solve(xs): return 1.001", contract["support_code"])
    assert not execute_public_checks(code, contract["checks"])["checks"][0]["satisfied"]


def test_nonfinite_or_boolean_outputs_are_rejected():
    contract = semantic_contract(POLYNOMIAL_PROBLEM)
    for value in ("float('nan')", "float('inf')", "True"):
        code = property_output(f"def solve(xs): return {value}", contract["support_code"])
        assert not execute_public_checks(code, contract["checks"])["checks"][0]["satisfied"]


def test_modified_public_polynomial_helper_does_not_activate_adapter():
    altered = POLYNOMIAL_PROBLEM.replace("math.pow(x, i)", "math.pow(x, i+1)")
    assert semantic_contract(altered)["mode"] == "literal_examples"


def test_renaming_public_function_does_not_change_semantic_rule():
    renamed = POLYNOMIAL_PROBLEM.replace("solve", "another_function")
    assert semantic_contract(renamed)["mode"] == "polynomial_root_residual"
    assert "another_function" in semantic_contract(renamed)["checks"][0]["expression"]


def test_conflicting_sort_example_causes_abstention_not_answer_rewrite():
    contract = semantic_contract(SORT_PROBLEM)
    assert contract["mode"] == "specification_conflict_abstention"
    assert not contract["checks"]
    assert contract["conflicts"][0]["rule_result"] == [1, 2, 4, 3, 5]


def test_negative_example_is_marked_outside_nonnegative_spec():
    problem = SORT_PROBLEM.replace("[1, 5, 2, 3, 4]", "[-2, -3]").replace(
        "[1, 2, 3, 4, 5]", "[-2, -3]"
    )
    contract = semantic_contract(problem)
    assert contract["conflicts"][0]["reason"] == "example_outside_nonnegative_domain"


def test_consistent_sort_example_keeps_literal_checks():
    problem = SORT_PROBLEM.replace("[1, 2, 3, 4, 5]", "[1, 2, 4, 3, 5]")
    assert semantic_contract(problem)["mode"] == "literal_examples"


def test_unrecognized_problem_is_not_claimed_semantically_supported():
    contract = semantic_contract("Calculate the minimum parallelogram area.")
    assert contract["mode"] == "unsupported"
    assert not contract["checks"]


def test_candidate_cannot_shadow_the_authored_property_function():
    contract = semantic_contract(POLYNOMIAL_PROBLEM)
    output = (
        "def solve(xs): return 10\n"
        "def _audit_public_root(value, coefficients): return True"
    )
    code = property_output(output, contract["support_code"])
    assert not execute_public_checks(code, contract["checks"])["checks"][0]["satisfied"]


def test_comparison_separates_abstention_from_matched_improvement():
    from experiments.audit_public_examples import compare_snapshots
    common = {
        "request_id": "same", "policy": "full", "task_id": "renamed", "repeat": 0,
        "benchmark_passed": True,
    }
    metadata = {"manifest_sha256": "manifest", "request_content_sha256": "requests"}
    literal = {**metadata, "outputs": [
        {**common, "public_status": "literal_contradiction"},
        {**common, "request_id": "dropped", "public_status": "literal_contradiction"},
    ]}
    semantic = {**metadata, "outputs": [
        {**common, "public_status": "examples_satisfied"},
    ]}
    report = compare_snapshots(literal, semantic)
    assert report["common_outputs"] == 1
    assert report["abstained_previously_covered_outputs"] == 1
    assert report["matched_by_policy"]["full"]["literal"]["covered_outputs"] == 1
    assert report["matched_by_policy"]["full"]["semantic"]["covered_outputs"] == 1
    assert len(report["changed_outcomes_on_common_outputs"]) == 1


def test_comparison_rejects_changed_source():
    import pytest
    from experiments.audit_public_examples import compare_snapshots
    with pytest.raises(ValueError, match="相同原始输入"):
        compare_snapshots({"manifest_sha256": "a"}, {"manifest_sha256": "b"})

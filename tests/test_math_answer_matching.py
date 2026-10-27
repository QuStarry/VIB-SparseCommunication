import copy
import importlib.util
from pathlib import Path

import pytest

from experiments.math_answer_matching import score


@pytest.mark.parametrize("prediction,reference,expected", [
    ("答案: 12,10,6", "12, 10, 6", True),
    ("答案: 6", "12, 10, 6", False),
    ("答案: 1/7", r"\frac{9}{7}", False),
    ("答案: 9/7", r"\dfrac{9}{7}", True),
    ("5.5", r"\frac{11}{2}", True),
    ("0.5", r"\frac12", True),
    ("25/16", r"\cfrac{25}{16}", True),
    ("{1}", "1", False),
    ("{1}", "{1}", True),
    ("x^{96}", "X^{96}", False),
    ("[1,2", "[1,2", False),
    ("1e9999999", "1", False),
    ("6+9i", "12+9i", False),
    ("7(x-3)(x+3)", "7(x+3)(x-3)", True),
    ("2*sqrt(3)", r"\sqrt{12}", True),
    (r"2\sqrt{3}", r"\sqrt{12}", True),
    ("3s²", "3s^2", True),
    ("(1, 4.5)", r"\left(1,\frac{9}{2}\right)", True),
    ("[1,2]", "[2,1]", False),
    ("{1,2}", "{2,1}", True),
    ("(-∞,-8) ∪ (8,∞)", r"(-\infty,-8)\cup(8,\infty)", True),
    ("(8,∞) ∪ (-∞,-8)", r"(-\infty,-8)\cup(8,\infty)", True),
    ("(-∞,0]", r"(-\infty,0)", False),
    ("答案: [-2,7]", r"x \in [-2,7]", True),
    ("b=4", "c=4", False),
    ("4", "b=4", True),
    ("$15,000", r"\$15,\!000", True),
    ("1,234.0", "1234", True),
    ("Evelyn", r"\text{Evelyn}", True),
    ("Leyevn", r"\text{Evelyn}", False),
    ("575", r"575\text{ students}", True),
    ("答案: 2", r"Proof: \boxed{2}, followed by graphics coordinates 5000", True),
    (r"\boxed{\frac{\sqrt{3}}{3}}", r"\frac{1}{\sqrt{3}}", True),
    ("Reasoning with a last number 7", "7", False),
    ("", "", False),
    ("__import__('os').system('exit 1')", "0", False),
])
def test_whole_answer_decisions(prediction, reference, expected):
    assert score(prediction, reference)["passed"] is expected


def test_rescoring_preserves_nonlabel_data():
    path = Path(__file__).resolve().parents[1] / "scripts/recompute_math_eval.py"
    spec = importlib.util.spec_from_file_location("recompute_math_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arm = {"reference_answer": "1,2", "output": "答案: 2", "passed": True,
           "history": ["unchanged"], "token_stats": {"total_tokens": 20}}
    vib_arm = copy.deepcopy(arm)
    vib_arm["output"] = "答案: 1,2"
    data = {
        "trace_results": {"math": [{"task_id": "MATH/test", "full": copy.deepcopy(arm),
                                   "vib": vib_arm}]},
        "detailed_results": {"math": [{"task_id": "MATH/test",
                                     "full_passed": True, "vib_passed": True,
                                     "full_messages": 3, "vib_messages": 2,
                                     "full_tokens": 20, "vib_tokens": 15}]},
        "summary": {"math": {"total_tasks": 1, "full_pass_rate": 100, "vib_pass_rate": 100,
                            "full_tokens": 20, "vib_tokens": 15}},
        "experiment_config": {"legacy": "preserved"},
    }
    before = module.invariant_digest(data)
    report = module.recompute(data)
    assert data["summary"]["math"]["full_pass_rate"] == 0
    assert data["summary"]["math"]["vib_pass_rate"] == 100
    assert data["summary"]["math"]["pass_delta_pp"] == 100
    assert report["analysis"]["full_pass_rate_pct"] == 0
    assert report["analysis"]["pass_delta_pp"] == 100
    assert module.invariant_digest(data) == before
    assert data["trace_results"]["math"][0]["full"]["output"] == "答案: 2"


def test_reference_coverage_is_independent_of_malformed_prediction():
    path = Path(__file__).resolve().parents[1] / "scripts/recompute_math_eval.py"
    spec = importlib.util.spec_from_file_location("recompute_math_coverage_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = {
        "trace_results": {"math": [{"task_id": "MATH/test",
            "full": {"reference_answer": "1", "output": "[1", "passed": False},
            "vib": {"reference_answer": "1", "output": "1", "passed": True},
        }]},
        "detailed_results": {"math": [{"task_id": "MATH/test", "full_passed": False,
            "vib_passed": True, "full_messages": 1, "vib_messages": 1,
            "full_tokens": 1, "vib_tokens": 1}]},
        "summary": {"math": {"total_tasks": 1, "full_pass_rate": 0, "vib_pass_rate": 100}},
    }
    report = module.recompute(data)
    assert all(row["reference_format"] == "expression" for row in report["rows"])
    assert report["reference_expression_sensitivity"]["total_tasks"] == 1
    assert report["reference_expression_sensitivity"]["pass_delta_pp"] == 100

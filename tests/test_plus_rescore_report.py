import pytest

from scripts.export_plus_rescore_report import POLICIES, coverage_counts, render, render_source_audit


def test_literal_input_counts_without_execution():
    tasks = {"HumanEval/0": {"test": "def check(candidate):\n    inputs = [[1], [2]]\n    return inputs"}}
    assert coverage_counts(tasks) == {"HumanEval/0": 2}


def test_dynamic_input_count_is_not_guessed():
    with pytest.raises(ValueError):
        coverage_counts({"HumanEval/0": {"test": "def check(candidate):\n    inputs = list(range(10))"}})


def test_dual_report_is_offline_and_retains_old_score():
    old = {policy: {"passed_generations": 32, "task_mean_accuracy_pct": 100.0} for policy in POLICIES}
    new = {policy: {"passed_generations": 30, "task_mean_accuracy_pct": 93.75} for policy in POLICIES}
    result = {
        "original_statistics": {"groups": old}, "statistics": {"groups": new},
        "primary_comparison": {"delta_pp": 0, "ci95_pp": [0, 0], "exact_sign_p": 1},
        "transitions": {policy: {"pass_to_fail": 2, "fail_to_pass": 0} for policy in POLICIES},
        "evaluations": [{"task_id": "HumanEval/0", "prompt_variant": policy,
                         "original_passed": True, "passed": False} for policy in POLICIES],
        "local_reference_evaluation_seconds": 1, "local_candidate_evaluation_seconds": 2,
    }
    manifest = {"task_ids": ["HumanEval/0"],
                "dataset": {"publisher": "evalplus/humanevalplus", "revision": "pinned",
                            "verified_publisher_lfs_sha256": "verified"}}
    html, markdown, changed = render(result, manifest, {"HumanEval/0": 1000})
    assert changed == ["HumanEval/0"]
    assert "32/32" in html and "30/32" in html
    assert "93.7500%" in html
    assert "{{" not in html
    assert "https://" not in html and "http://" not in html
    assert 'data-track="enhanced"' in html
    assert "只看标签变化题" in html
    assert "不是新预注册确认性实验" in markdown
    assert "新增模型请求0" in markdown


def test_failed_source_report_does_not_invent_candidate_scores():
    audit = {"passed_references": 163,
             "failed_references": [{"task_id": "HumanEval/32", "status": "runtime_exception"}],
             "independent_reference_executions": 328, "local_reference_evaluation_seconds": 42.0}
    manifest = {"task_ids": ["HumanEval/0"],
                "dataset": {"publisher": "evalplus/humanevalplus", "revision": "pinned",
                            "verified_publisher_lfs_sha256": "verified"}}
    html, markdown = render_source_audit(audit, manifest, {"HumanEval/0": 1000})
    assert "163/164" in html
    assert "128份未重评" in html
    assert "未执行" in html
    assert "候选增强评分未执行" in markdown
    assert "不是VIB候选失败" in markdown
    assert "https://" not in html and "http://" not in html

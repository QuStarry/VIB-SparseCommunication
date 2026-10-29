from scripts.export_repaired_plus_report import render_local
from scripts.export_plus_rescore_report import POLICIES


def test_local_report_discloses_both_numerical_and_reference_revisions():
    groups = {policy: {"passed_generations": 32, "task_mean_accuracy_pct": 100} for policy in POLICIES}
    result = {
        "original_statistics": {"groups": groups}, "statistics": {"groups": groups},
        "primary_comparison": {"delta_pp": 0, "ci95_pp": [0, 0], "exact_sign_p": 1},
        "transitions": {policy: {"pass_to_fail": 0, "fail_to_pass": 0} for policy in POLICIES},
        "evaluations": [{"task_id": "HumanEval/0", "prompt_variant": policy,
                         "original_passed": True, "passed": True} for policy in POLICIES],
        "local_reference_evaluation_seconds": 1, "local_candidate_evaluation_seconds": 2,
    }
    manifest = {"task_ids": ["HumanEval/0"], "patch": {"local_file_sha256": "local-digest"},
                "dataset": {"publisher": "evalplus/humanevalplus", "revision": "pinned",
                            "verified_publisher_lfs_sha256": "publisher-digest"}}
    page, markdown, changed = render_local(result, manifest, {"HumanEval/0": 1000})
    assert changed == []
    assert "本地修订增强重评" in page
    assert "不是官方EvalPlus成绩" in page
    assert "886组预期根表" in page
    assert "相邻浮点" in markdown
    assert "32不在候选16题中" in markdown
    assert "163个原canonical与32题发布方根表" in markdown
    assert "local_repaired_test.jsonl" in markdown
    assert "{{" not in page

from experiments.audit_public_examples import (
    classify_evaluation, diagnostic_metrics, extract_examples,
)


def test_extracts_multiline_and_inline_literal_examples():
    problem = "def foo(x):\n    >>> foo([1, 2])\n    [2, 3]\n    >>> foo(1) == 2"
    extracted = extract_examples(problem)
    assert [row["expected"] for row in extracted["checks"]] == [[2, 3], 2]
    assert not extracted["rejected_examples"]


def test_nested_round_and_negative_result_are_supported():
    problem = "def foo(x):\n    >>> round(foo([1, 2]), 2) # comment\n    -0.5"
    checks = extract_examples(problem)["checks"]
    assert checks[0]["expression"] == "round(foo([1, 2]), 2)"
    assert checks[0]["expected"] == -0.5


def test_malformed_sort_example_is_not_repaired():
    problem = "def foo(x):\n    >>> foo([1, 2]) [1, 2]\n"
    result = extract_examples(problem)
    assert not result["checks"]
    assert len(result["rejected_examples"]) == 1


def test_conflicting_identical_inputs_abstain():
    problem = "def foo(x):\n    >>> foo(1) == 2\n    >>> foo(1) == 3"
    result = extract_examples(problem)
    assert not result["checks"]
    assert result["conflicting_expressions"] == ["foo(1)"]


def test_dangerous_expressions_are_never_evaluated_during_extraction():
    problem = (
        "def foo(x):\n"
        "    >>> open('/tmp/should-not-exist', 'w')\n    1\n"
        "    >>> foo(__import__('os').system('exit 1'))\n    1\n"
        "    >>> foo(1)\n    __import__('os').system('exit 1')"
    )
    result = extract_examples(problem)
    assert not result["checks"]
    assert len(result["rejected_examples"]) == 3


def test_no_literal_examples_is_explicitly_uncovered():
    assert extract_examples("Input n=5; Output 1")["checks"] == []


def test_non_json_literal_types_abstain_without_false_alarms():
    result = extract_examples("def foo(x):\n    >>> foo(1)\n    (1, 2)")
    assert not result["checks"]
    assert len(result["rejected_examples"]) == 1


def test_runtime_errors_do_not_count_as_literal_contradictions():
    evaluation = {"status": "executed", "checks": [
        {"error_type": "NameError", "satisfied": False},
    ]}
    assert classify_evaluation(evaluation) == "execution_error"
    assert classify_evaluation({"status": "timeout"}) == "execution_error"


def test_false_alarm_denominator_is_covered_passed_outputs():
    rows = [
        {"public_status": "literal_contradiction", "benchmark_passed": True},
        {"public_status": "examples_satisfied", "benchmark_passed": True},
        {"public_status": "literal_contradiction", "benchmark_passed": False},
        {"public_status": "examples_satisfied", "benchmark_passed": False},
        {"public_status": "execution_error", "benchmark_passed": False},
    ]
    metrics = diagnostic_metrics(rows)
    assert metrics["false_alarm_rate_on_covered_passed"] == 0.5
    assert metrics["failure_detection_rate_on_covered_failed"] == 1 / 3
    assert metrics["benchmark_failures_despite_satisfied_examples"] == 1
    assert metrics["execution_error_outputs"] == 1
    assert diagnostic_metrics([])["false_alarm_rate_on_covered_passed"] is None


def test_legacy_math_manifest_without_dataset_abstains(tmp_path, monkeypatch):
    import json
    from experiments import audit_public_examples as module
    monkeypatch.setattr(module, "analyze", lambda path: None)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "tasks": [{"task_id": "MATH/1", "problem": "public math"}],
        "policies": ["full"],
    }))
    report = module.audit_examples(tmp_path)
    assert report["coverage"]["dataset"] == "math"
    assert report["coverage"]["tasks_with_checks"] == 0
    assert report["performance"]["execution_elapsed_ms_including_process_startup"] == {}

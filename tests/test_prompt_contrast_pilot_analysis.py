import json
import os
from pathlib import Path

import pytest

from experiments.analyze_prompt_contrast_pilot import analyze_pilot, paired_statistics
from experiments.run_prompt_contrast_pilot import execute_pilot, freeze_pilot
from tests.test_prompt_contrast_pilot import ENDPOINT, SOURCE, factory


@pytest.fixture(autouse=True)
def fake_disk_flush(monkeypatch):
    monkeypatch.setattr(os, "fsync", lambda descriptor: None)


def test_bootstrap_uses_task_means_not_independent_repetitions():
    rows = [
        {"task_id": task, "prompt_variant": variant, "repetition": repeat,
         "passed": variant == "a" and task == "one"}
        for task in ("one", "two") for variant in ("a", "b") for repeat in (0, 1)]
    stats = paired_statistics(rows, samples=10000, seed=20261028)
    assert stats["groups"]["a"]["passed_generations"] == 2
    assert stats["groups"]["a"]["task_mean_accuracy_pct"] == 50
    contrast = stats["comparisons"][0]
    assert contrast["left"] == "a" and contrast["right"] == "b"
    assert contrast["delta_pp"] == 50
    assert contrast["ci95_pp"] == [0, 100]
    assert contrast["independent_tasks"] == 2


def test_incomplete_pairs_are_rejected():
    with pytest.raises(ValueError):
        paired_statistics([{"task_id": "one", "prompt_variant": "a",
                            "repetition": 0, "passed": True}])


def test_scoring_only_after_generation_complete_and_no_model_on_resume(tmp_path, monkeypatch):
    from experiments import analyze_prompt_contrast_pilot as analysis
    calls = []
    manifest = freeze_pilot(SOURCE, ENDPOINT)
    output = tmp_path / "pilot"
    execute_pilot(manifest, output, factory(calls))
    evaluations = []

    def scorer(task):
        def evaluate(content):
            assert len(calls) == 144
            evaluations.append(task)
            return False
        return evaluate

    monkeypatch.setattr(analysis, "load_evaluators",
                        lambda *args: {task: scorer(task) for task in manifest["task_ids"]})
    report = analyze_pilot(output)
    assert len(evaluations) == 144
    assert all(group["passed_generations"] == 0 for group in report["statistics"]["groups"].values())
    assert (output / "README.md").exists()
    assert analyze_pilot(output) == report
    assert len(evaluations) == 144
    assert len(calls) == 144


def test_truncated_is_failure_without_invoking_hidden_tests(tmp_path, monkeypatch):
    from experiments import analyze_prompt_contrast_pilot as analysis
    output = tmp_path / "pilot"
    manifest = freeze_pilot(SOURCE, ENDPOINT)
    execute_pilot(manifest, output, factory([], finish="length", completion_tokens=1024))
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: {
        task: lambda text: pytest.fail("truncated must not be scored") for task in manifest["task_ids"]})
    result = analyze_pilot(output)
    assert all(row["passed"] is False and row["status"] == "truncated" for row in result["evaluations"])


def test_unknown_generation_cannot_be_analyzed(tmp_path, monkeypatch):
    from experiments import analyze_prompt_contrast_pilot as analysis
    output = tmp_path / "pilot"
    execute_pilot(freeze_pilot(SOURCE, ENDPOINT), output, factory([], missing_usage=True))
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: pytest.fail("incomplete"))
    with pytest.raises(ValueError):
        analyze_pilot(output)


def test_raw_tampering_cannot_change_score(tmp_path):
    output = tmp_path / "pilot"
    execute_pilot(freeze_pilot(SOURCE, ENDPOINT), output, factory([]))
    path = next((output / "responses").glob("*.json"))
    raw = json.loads(path.read_text())
    raw["content"] = "modified"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        analyze_pilot(output)


def test_actual_isolated_scorer_handles_frozen_official_solution(tmp_path):
    from src.isolated_code_evaluation import load_evaluators
    from src.experiment_driver import file_sha256
    bundle = SOURCE.parent / "evaluation.json"
    data = json.loads(bundle.read_text())
    task = data["tasks"][0]
    evaluator = load_evaluators(bundle, file_sha256(bundle))[task["task_id"]]
    assert evaluator("def has_close_elements(numbers, threshold):\n"
                     "    return any(abs(a - b) < threshold for i, a in enumerate(numbers)\n"
                     "               for b in numbers[i + 1:])")

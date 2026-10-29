import os

import pytest

from experiments.analyze_gate_receiver import analyze_gate_session
from experiments.run_gate_receiver import execute_gate_session
from tests.test_gate_receiver_session import freeze
from tests.test_prompt_contrast_pilot import factory


@pytest.fixture(autouse=True)
def fake_flush(monkeypatch):
    monkeypatch.setattr(os, "fsync", lambda descriptor: None)


def test_scoring_after_all_requests_and_primary_message_budget_report(tmp_path, monkeypatch):
    from experiments import analyze_gate_receiver as analysis
    manifest, output, calls, scores = freeze(), tmp_path / "run", [], []
    execute_gate_session(manifest, output, factory(calls))

    def evaluate(content):
        assert len(calls) == 256
        scores.append(content)
        return True
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: {
        task: evaluate for task in manifest["task_ids"]})
    result = analyze_gate_session(output)
    assert len(scores) == 256
    assert result["primary_comparison"]["left"] == "vib"
    assert result["primary_comparison"]["right"] == "random_matched"
    assert result["communication"]["vib"]["messages"] == 129
    assert result["communication"]["random_matched"]["messages"] == 129
    assert result["communication"]["full"]["messages"] == 192
    assert result["communication"]["vib"]["matched_per_task_per_round"]
    assert analyze_gate_session(output) == result
    assert len(scores) == 256


def test_incomplete_generation_cannot_produce_gate_quality(tmp_path, monkeypatch):
    from experiments import analyze_gate_receiver as analysis
    output = tmp_path / "run"
    execute_gate_session(freeze(), output, factory([], missing_usage=True))
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: pytest.fail("incomplete"))
    with pytest.raises(ValueError):
        analyze_gate_session(output)


def test_truncated_retained_as_failure_without_scoring(tmp_path, monkeypatch):
    from experiments import analyze_gate_receiver as analysis
    manifest, output = freeze(), tmp_path / "run"
    execute_gate_session(manifest, output, factory([], finish="length", completion_tokens=1024))
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: {
        task: lambda content: pytest.fail("truncated") for task in manifest["task_ids"]})
    result = analyze_gate_session(output)
    assert not any(row["passed"] for row in result["evaluations"])


def test_scoring_facility_error_does_not_regenerate(tmp_path, monkeypatch):
    from experiments import analyze_gate_receiver as analysis
    manifest, output, calls = freeze(), tmp_path / "run", []
    execute_gate_session(manifest, output, factory(calls))

    def broken(content):
        raise RuntimeError("Facility failure")
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: {
        task: broken for task in manifest["task_ids"]})
    with pytest.raises(RuntimeError):
        analyze_gate_session(output)
    assert len(calls) == 256
    assert not (output / "analysis.json").exists()

import json

import pytest

from experiments.analyze_feedback_experiment import analyze_feedback
from experiments.run_feedback_experiment import run_feedback
from tests.test_feedback_experiment import fake_pipeline, fast_flush
from tests.test_prompt_contrast_pilot import factory


def test_actual_and_standalone_costs_not_double_counted(tmp_path, fake_pipeline, monkeypatch):
    import experiments.analyze_feedback_experiment as analysis

    output = tmp_path / "pipeline"
    run_feedback(output, factory([]))
    monkeypatch.setattr(analysis, "load_evaluators",
                        lambda path, digest: {f"HumanEval/{index}": lambda code: True for index in range(164)})
    result = analyze_feedback(output)
    assert result["actual_experiment_cost"]["requests"] == 368
    assert result["actual_experiment_cost"]["prompt_tokens"] == 36800
    assert result["actual_experiment_cost"]["proxy_usd"] == "0.019872"
    assert result["statistics"]["groups"]["vib"]["passed_generations"] == 32
    for policy in result["policy_costs"].values():
        assert policy["standalone_requests"] == 112
        assert policy["standalone_one_synthesis_proxy_usd"] == "0.006048"
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: pytest.fail("cached scoring"))
    assert analyze_feedback(output) == result
    path = output / "analysis.json"
    corrupted = json.loads(path.read_text())
    corrupted["primary_comparison"]["delta_pp"] = 100
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError):
        analyze_feedback(output)


def test_incomplete_never_loads_hidden_scoring(tmp_path, fake_pipeline, monkeypatch):
    import experiments.analyze_feedback_experiment as analysis

    output = tmp_path / "pipeline"
    run_feedback(output, factory([], missing_usage=True))
    monkeypatch.setattr(analysis, "load_evaluators", lambda *args: pytest.fail("hidden"))
    with pytest.raises(ValueError):
        analyze_feedback(output)


def test_final_truncation_fails_without_scoring_or_regeneration(tmp_path, fake_pipeline, monkeypatch):
    import experiments.analyze_feedback_experiment as analysis

    calls = []
    output = tmp_path / "pipeline"
    run_feedback(output, factory(calls, finish="length"))
    monkeypatch.setattr(analysis, "load_evaluators",
                        lambda *args: {f"HumanEval/{index}": lambda code: pytest.fail("truncated score")
                                       for index in range(164)})
    result = analyze_feedback(output)
    assert len(calls) == 368
    assert result["truncated_by_phase"] == {"round1": 48, "round2": 192, "synthesis": 128}
    assert all(not row["passed"] for row in result["evaluations"])
    assert result["actual_experiment_cost"]["proxy_usd"] == "0.019872"

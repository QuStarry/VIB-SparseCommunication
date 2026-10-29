import json
import random
from types import SimpleNamespace

import pytest

from src.isolated_code_evaluation import (
    evaluation_protocol, load_evaluators, score_code,
)
from src.experiment_driver import file_sha256


def task():
    return {"task_id": "HumanEval/test", "entry_point": "f",
            "test_cases": ["def check(candidate):\n    assert candidate(1) == 2"]}


def test_check_is_called_not_merely_defined():
    assert score_code("def f(x): return x + 1", task()) is True
    assert score_code("def f(x): return x", task()) is False


def test_fenced_code_is_sanitized_consistently():
    assert score_code("```python\ndef f(x): return x+1\n```", task())


def test_timeout_does_not_count_as_pass():
    assert not score_code("while True: pass", task(), timeout=0.3)


def test_system_exit_cannot_fake_success():
    assert not score_code("raise SystemExit(42)", task())


def test_unexpected_process_exit_is_not_silently_an_algorithm_failure():
    with pytest.raises(RuntimeError, match="protocol"):
        score_code("import os\nos._exit(0)", task())


def test_stdout_cannot_spoof_result():
    assert not score_code('print("true")\ndef f(x): return x', task())


def test_child_does_not_inherit_parent_credentials(monkeypatch):
    monkeypatch.setenv("PRIVATE_API_KEY_1", "fake-secret")
    code = 'import os\nassert "PRIVATE_API_KEY_1" not in os.environ\ndef f(x): return x+1'
    assert score_code(code, task())


def test_child_random_seed_does_not_pollute_parent_state():
    original = random.getstate()
    assert score_code("import random\nrandom.seed(123)\ndef f(x): return x+1", task())
    assert random.getstate() == original


def test_child_working_directory_is_temporary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert score_code('open("child.txt", "w").write("test")\ndef f(x): return x+1', task())
    assert not (tmp_path / "child.txt").exists()


def test_inconsistent_independent_runs_are_rejected(monkeypatch):
    from src import isolated_code_evaluation as evaluation
    results = iter([True, False])
    monkeypatch.setattr(evaluation, "_score_once", lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match="inconsistent"):
        score_code("def f(x): return x+1", task())


def test_bundle_binds_task_set_and_score_protocol(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({**evaluation_protocol(), "tasks": [task()]}), encoding="utf-8")
    evaluators = load_evaluators(path, file_sha256(path))
    assert set(evaluators) == {"HumanEval/test"}
    assert evaluators["HumanEval/test"]("def f(x): return x+1")
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        evaluators["HumanEval/test"]("def f(x): return x+1")


def test_bundle_source_drift_is_rejected(tmp_path):
    path = tmp_path / "evaluation.json"
    bundle = {**evaluation_protocol(), "tasks": [task()]}
    bundle["implementation_sha256"] = "0" * 64
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation"):
        load_evaluators(path, file_sha256(path))


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError):
        score_code("pass", task(), timeout=timeout)


def test_duplicate_task_ids_in_bundle_are_rejected(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({**evaluation_protocol(), "tasks": [task(), task()]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_evaluators(path, file_sha256(path))


def test_received_session_resumes_with_real_isolated_evaluator(tmp_path):
    from src.experiment_session import ExperimentSession

    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({**evaluation_protocol(), "tasks": [task()]}), encoding="utf-8")
    evaluator = load_evaluators(path, file_sha256(path))["HumanEval/test"]
    manifest = {
        "version": 1, "provider": "custom", "model": "test-model",
        "base_url": "https://example.invalid/v1", "budget_usd": "0.001",
        "evaluation_version": evaluation_protocol()["version"],
        "price": {"version": "synthetic-v1", "input_usd_per_million": "1",
                  "output_usd_per_million": "1", "scope": "proxy_not_invoice"},
        "jobs": [{"job_id": "job/0", "task_id": "HumanEval/test", "stage": "code",
                  "messages": [{"role": "user", "content": "Return x+1."}],
                  "parameters": {"temperature": 0, "max_tokens": 100}, "prompt_token_limit": 100}],
    }
    requests = []

    def fake_attempt(*args, **kwargs):
        requests.append(kwargs)
        return {"content": "def f(x): return x+1", "usage_status": "known",
                "prompt_tokens": 10, "completion_tokens": 10, "served_model": "test-model",
                "finish_reason": "stop", "response_id": "test-response"}

    client = SimpleNamespace(provider="custom", model="test-model", base_url="https://example.invalid/v1",
                             chat_single_attempt=fake_attempt)
    database = tmp_path / "session.db"
    session = ExperimentSession(database, manifest)
    session.execute("job/0", client)
    restored = ExperimentSession(database, manifest)
    assert restored.evaluate("job/0", evaluator, manifest["evaluation_version"])
    assert restored.summary()["run_complete"]
    assert len(requests) == 1

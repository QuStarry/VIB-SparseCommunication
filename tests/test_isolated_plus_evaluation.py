import pytest

from src.isolated_plus_evaluation import plus_protocol, score_plus


def task(test="def check(candidate):\n    assert candidate(3) == 6"):
    return {"entry_point": "f", "test": test}


def test_correct_code_passes_twice():
    result = score_plus("def f(x): return x * 2", task())
    assert result == {"passed": True, "status": "pass", "independent_checks": 2}


def test_wrong_code_fails_expanded_checks():
    assert score_plus("def f(x): return x", task())["status"] == "assertion_failure"


def test_numpy_available_without_site_initialization():
    tests = "import numpy as np\ndef check(candidate):\n    assert np.allclose(candidate(3), [6.0])"
    assert score_plus("def f(x): return [float(x * 2)]", task(tests))["passed"]


def test_candidate_does_not_inherit_api_credentials(monkeypatch):
    monkeypatch.setenv("PRIVATE_API_KEY_1", "test-secret-never-inherited")
    code = "import os\ndef f(x):\n    assert 'PRIVATE_API_KEY_1' not in os.environ\n    return x * 2"
    assert score_plus(code, task())["passed"]


def test_exception_is_distinct_from_assertion():
    assert score_plus("def f(x): raise ValueError('x')", task())["status"] == "runtime_exception"


def test_timeout_is_failure_without_retry():
    assert score_plus("def f(x):\n    while True: pass", task(), timeout=1)["status"] == "timeout"


def test_code_fence_extraction_does_not_repair():
    assert score_plus("```python\ndef f(x): return x * 2\n```", task())["passed"]


def test_unexpected_exit_is_infrastructure_error():
    with pytest.raises(RuntimeError, match="outside"):
        score_plus("import os\nos._exit(0)", task())


def test_two_interpreter_disagreement_is_not_voting(monkeypatch):
    import src.isolated_plus_evaluation as module
    statuses = iter(["pass", "assertion_failure"])
    monkeypatch.setattr(module, "_once", lambda *args: next(statuses))
    with pytest.raises(RuntimeError, match="inconsistent"):
        score_plus("def f(x): return x * 2", task())


def test_protocol_explains_runtime_and_timeout_scope():
    protocol = plus_protocol()
    assert protocol["independent_checks"] == 2
    assert protocol["site_initialization"] is False
    assert protocol["numpy"]["version"]
    assert protocol["timeout_scope"] == "overall_worker_wall_time_not_evalplus_reference_scaled_per_input"

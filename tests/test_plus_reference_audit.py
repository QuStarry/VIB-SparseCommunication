import pytest

import experiments.audit_plus_reference_suite as audit
from experiments.rescore_feedback_plus import prepare_plus
from tests.test_feedback_plus_rescoring import synthetic_source


def test_full_reference_audit_does_not_authorize_candidates(synthetic_source, monkeypatch):
    source, output = synthetic_source
    prepare_plus(source, output)
    calls = []
    def score(code, task, manifest):
        calls.append(code)
        failed = len(calls) == 3
        return {"passed": not failed, "status": "runtime_exception" if failed else "pass",
                "independent_checks": 2}
    monkeypatch.setattr(audit, "_score", score)
    result = audit.audit_reference_suite(output)
    assert len(calls) == 164
    assert result["passed_references"] == 163
    assert len(result["failed_references"]) == 1
    assert result["candidate_outputs_evaluated"] == 0
    assert result["candidate_rescoring_authorized"] is False
    assert result["new_model_requests"] == 0
    assert not (output / "candidate_checks").exists()
    assert not (output / "analysis.json").exists()
    monkeypatch.setattr(audit, "_score", lambda *args: pytest.fail("cached source audit"))
    assert audit.audit_reference_suite(output) == result


def test_source_audit_obeys_existing_rescore_lock(synthetic_source):
    source, output = synthetic_source
    prepare_plus(source, output)
    (output / ".rescore.lock").touch()
    with pytest.raises(FileExistsError):
        audit.audit_reference_suite(output)
    assert (output / ".rescore.lock").exists()

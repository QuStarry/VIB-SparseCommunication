import json

import pytest

import experiments.rescore_feedback_plus as base
import experiments.rescore_repaired_plus as repaired
from src.experiment_driver import file_sha256
from tests.test_feedback_plus_rescoring import synthetic_source
from tests.test_repaired_plus import task32


@pytest.fixture
def local_session(synthetic_source, monkeypatch):
    source, parent = synthetic_source
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[32] = {**task32(), "prompt": "def find_zero(xs):\n",
                "canonical_solution": "    return 2.0\n"}
    source.write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(base, "SOURCE_SHA256", file_sha256(source))
    base.prepare_plus(source, parent)
    monkeypatch.setattr(repaired, "PARENT", parent)
    monkeypatch.setattr(repaired, "_score", lambda *args: {
        "passed": True, "status": "pass", "independent_checks": 2})
    return parent, parent.parent / "repaired"


def test_local_patch_preserves_original_and_other_163_tasks(local_session):
    parent, output = local_session
    before = (parent / "publisher_test.jsonl").read_bytes()
    manifest = repaired.prepare_repaired(output)
    after = (parent / "publisher_test.jsonl").read_bytes()
    assert before == after
    local = [json.loads(line) for line in (output / "local_repaired_test.jsonl").read_text().splitlines()]
    original = [json.loads(line) for line in before.decode().splitlines()]
    assert [index for index in range(164) if original[index] != local[index]] == [32]
    assert local[32]["canonical_solution"] == original[32]["canonical_solution"]
    assert manifest["patch"]["numerical_acceptance_extended"]
    assert manifest["patch"]["reference_validation_override"]["remaining_original_canonicals"] == 163
    assert manifest["patch"]["task32_in_candidate_cohort"] is False


def test_complete_local_rescore_and_cache(local_session, monkeypatch):
    _, output = local_session
    repaired.prepare_repaired(output)
    result = repaired.rescore_repaired(output)
    assert len(result["evaluations"]) == 128
    assert result["reference_validation_scope"] == "163_original_canonicals_plus_task32_publisher_root_table"
    assert result["independent_candidate_executions"] == 256
    assert result["new_model_requests"] == 0
    monkeypatch.setattr(repaired, "_score", lambda *args: pytest.fail("cached candidates"))
    monkeypatch.setattr(base, "_score", lambda *args: pytest.fail("cached references"))
    assert repaired.rescore_repaired(output) == result


def test_reference_failure_still_blocks_candidates(local_session, monkeypatch):
    _, output = local_session
    repaired.prepare_repaired(output)
    monkeypatch.setattr(base, "_score", lambda *args: {
        "passed": False, "status": "assertion_failure", "independent_checks": 2})
    with pytest.raises(ValueError, match="reference failed"):
        repaired.rescore_repaired(output)
    assert not (output / "candidate_checks").exists()


def test_patch_asset_drift_is_not_repaired_silently(local_session):
    _, output = local_session
    repaired.prepare_repaired(output)
    (output / "local_repaired_test.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="dataset changed"):
        repaired.rescore_repaired(output)


def test_analysis_tamper_rejected(local_session):
    _, output = local_session
    repaired.prepare_repaired(output)
    repaired.rescore_repaired(output)
    path = output / "analysis.json"
    payload = json.loads(path.read_text())
    payload["statistics"]["groups"]["vib"]["passed_generations"] = 100
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Cached"):
        repaired.rescore_repaired(output)

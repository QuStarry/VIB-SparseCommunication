import copy
import json
import os
from pathlib import Path
import socket

import pytest

from experiments.check_prompt_contrast_readiness import (
    endpoint_metadata, planning_scenario, convert_draft, check_readiness,
)
from src.experiment_driver import driver_sha256, file_sha256, implementation_sha256
from src.isolated_code_evaluation import evaluation_protocol
from src.request_ledger import fingerprint


def config(tmp_path):
    path = tmp_path / "pro.env"
    path.write_text(
        "PRIVATE_API_URL=https://example.invalid/v1\n"
        "PRIVATE_API_KEY=exhausted-secret\nPRIVATE_API_KEY_1=second-secret\n"
        "模型名称\tModel id\ndeepseek-v4-flash\tdeepseek-v4-flash-0817\n", encoding="utf-8",
    )
    return path


def preparation(tmp_path):
    endpoint = {"provider": "private", "model": "deepseek-v4-flash-0817", "base_url": "https://example.invalid/v1"}
    bundle = tmp_path / "evaluation.json"
    bundle.write_text(json.dumps({**evaluation_protocol(), "tasks": [
        {"task_id": "task/0", "entry_point": "f", "test_cases": ["def check(c): assert c()==1"]}
    ]}), encoding="utf-8")
    snapshots = {}
    for variant in ["legacy_compact", "current_arbitration", "specification_priority"]:
        path = tmp_path / f"{variant}.py"
        path.write_text("# Synthetic test-only prompt snapshot\n", encoding="utf-8")
        snapshots[variant] = {"path": path.name, "sha256": file_sha256(path)}
    jobs = [
        {"job_id": f"task/0/{variant}/{repeat}", "task_id": "task/0", "stage": "code",
         "prompt_variant": variant, "repetition": repeat, "candidate_stream_sha256": "test-stream",
         "messages": [{"role": "user", "content": "Return one."}],
         "parameters": {"temperature": 0, "max_tokens": 1024}, "prompt_token_limit": None}
        for variant in snapshots for repeat in range(2)
    ]
    for job in jobs:
        job["messages_sha256"] = fingerprint(job["messages"])
    draft = {
        "version": "archival_prompt_contrast_draft_v1", "status": "not_authorized",
        "task_count": 1, "repeats": 2, "jobs": jobs, "prompt_sources": snapshots,
        "evaluation": {"path": bundle.name, "sha256": file_sha256(bundle), "version": evaluation_protocol()["version"]},
        "builder_sha256": file_sha256(Path(__file__).resolve().parents[1] / "experiments/prepare_prompt_contrast.py"),
        "design": "exploratory", "budget": {"proposed_usd": "3", "approved": False},
    }
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(draft), encoding="utf-8")
    (tmp_path / "preparation_report.json").write_text(json.dumps({"draft_sha256": file_sha256(path)}), encoding="utf-8")
    proof = tmp_path / "proof.txt"
    proof.write_text("Synthetic reviewed evidence, not production pricing.", encoding="utf-8")
    artifact = {"path": str(proof), "sha256": file_sha256(proof)}
    price = {"version": "synthetic-v1", "input_usd_per_million": "0.3",
             "output_usd_per_million": "1.2", "scope": "proxy_not_invoice"}
    evidence = {
        "draft_sha256": file_sha256(path), "endpoint": endpoint, "price": price, "budget_usd": "3",
        "preflight": {
            "driver_sha256": driver_sha256(), "implementation_sha256": implementation_sha256(),
            "price_evidence": {**artifact, "endpoint": endpoint, "price": price, "reviewed_upper_bound": True},
            "evaluation_evidence": {"path": str(bundle), "sha256": file_sha256(bundle),
                                    "version": draft["evaluation"]["version"]},
            "prompt_bounds": {job["job_id"]: {**artifact, "endpoint": endpoint,
                "messages_sha256": job["messages_sha256"], "prompt_token_limit": 10, "max_tokens": 1024,
                "reviewed_provider_bounds": True} for job in jobs},
        },
    }
    return path, draft, endpoint, evidence


def test_config_selects_second_key_without_exporting_or_hashing_credentials(tmp_path):
    before = dict(os.environ)
    result = endpoint_metadata(config(tmp_path))
    assert result["credential_env"] == "PRIVATE_API_KEY_1"
    assert result["credential_present"]
    assert result["model"] == "deepseek-v4-flash-0817"
    assert "second-secret" not in json.dumps(result)
    assert "exhausted-secret" not in json.dumps(result)
    assert "api_key" not in result
    assert dict(os.environ) == before


def test_missing_second_key_does_not_fall_back(tmp_path):
    path = config(tmp_path)
    path.write_text(path.read_text().replace("PRIVATE_API_KEY_1=second-secret\n", ""), encoding="utf-8")
    assert endpoint_metadata(path)["credential_present"] is False


def test_endpoint_url_with_credentials_is_rejected(tmp_path):
    path = config(tmp_path)
    path.write_text(path.read_text().replace("https://example.invalid", "https://user:secret@example.invalid"), encoding="utf-8")
    with pytest.raises(ValueError):
        endpoint_metadata(path)


def test_unlisted_model_id_is_not_assumed_valid(tmp_path):
    with pytest.raises(ValueError):
        endpoint_metadata(config(tmp_path), model="unknown")


def test_config_resolution_matches_existing_private_resolver(tmp_path, monkeypatch):
    from src.llm_client import _resolve_provider
    monkeypatch.setenv("VIB_PRIVATE_API_KEY_ENV", "PRIVATE_API_KEY_1")
    path = config(tmp_path)
    expected = _resolve_provider("private", "deepseek-v4-flash", str(path))
    actual = endpoint_metadata(path)
    assert (actual["provider"], actual["model"], actual["base_url"]) == (
        expected.provider, expected.model_id, expected.base_url,
    )


def test_credential_variable_argument_cannot_be_a_key_value(tmp_path):
    with pytest.raises(ValueError):
        endpoint_metadata(config(tmp_path), credential_env="second-secret")


def test_planning_bytes_do_not_become_verified_tokens(tmp_path):
    _, draft, _, _ = preparation(tmp_path)
    report = planning_scenario(draft)
    assert not report["guaranteed_budget_compliance"]
    assert report["prompt_tokens"] is None
    assert report["output_cap_only_historical_proxy_usd"] == "0.0073728"
    assert all(job["prompt_token_limit"] is None for job in draft["jobs"])


def test_no_evidence_means_no_executable_candidate(tmp_path):
    path, _, endpoint, _ = preparation(tmp_path)
    candidate, report = convert_draft(path, endpoint, None)
    assert candidate is None
    assert not report["ready"]


def test_complete_synthetic_evidence_preserves_jobs_and_still_requires_approval(tmp_path):
    path, draft, endpoint, evidence = preparation(tmp_path)
    candidate, report = convert_draft(path, endpoint, evidence)
    assert report["ready"]
    assert len(candidate["jobs"]) == len(draft["jobs"])
    assert [job["messages"] for job in candidate["jobs"]] == [job["messages"] for job in draft["jobs"]]
    assert "approval" not in candidate
    assert report["paid_authorized"] is False
    assert all(job["prompt_token_limit"] == 10 for job in candidate["jobs"])


@pytest.mark.parametrize("field", ["draft_sha256", "endpoint"])
def test_evidence_must_bind_original_draft_and_endpoint(tmp_path, field):
    path, _, endpoint, evidence = preparation(tmp_path)
    evidence[field] = "changed"
    assert convert_draft(path, endpoint, evidence)[0] is None


def test_evaluation_cannot_be_silently_replaced(tmp_path):
    path, _, endpoint, evidence = preparation(tmp_path)
    alternate = tmp_path / "other.json"
    alternate.write_text("{}", encoding="utf-8")
    evidence["preflight"]["evaluation_evidence"]["path"] = str(alternate)
    assert convert_draft(path, endpoint, evidence)[0] is None


def test_incomplete_prompt_bound_list_is_rejected(tmp_path):
    path, _, endpoint, evidence = preparation(tmp_path)
    evidence["preflight"]["prompt_bounds"].pop(next(iter(evidence["preflight"]["prompt_bounds"])))
    assert convert_draft(path, endpoint, evidence)[0] is None


def test_budget_cannot_silently_select_a_subset(tmp_path):
    path, _, endpoint, evidence = preparation(tmp_path)
    evidence["budget_usd"] = "0.00001"
    candidate, report = convert_draft(path, endpoint, evidence)
    assert candidate is None
    assert "budget_cannot_cover_all_reservations" in report["blockers"]


def test_draft_artifact_mutation_is_rejected(tmp_path):
    path, _, endpoint, evidence = preparation(tmp_path)
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        convert_draft(path, endpoint, evidence)


def test_prompt_snapshot_mutation_is_rejected(tmp_path):
    path, draft, endpoint, evidence = preparation(tmp_path)
    (tmp_path / draft["prompt_sources"]["legacy_compact"]["path"]).write_text("# Changed", encoding="utf-8")
    with pytest.raises(ValueError):
        convert_draft(path, endpoint, evidence)


def test_unreviewed_bounds_cannot_be_promoted(tmp_path):
    path, _, endpoint, evidence = preparation(tmp_path)
    proof = next(iter(evidence["preflight"]["prompt_bounds"].values()))
    proof["reviewed_provider_bounds"] = False
    assert convert_draft(path, endpoint, evidence)[0] is None


def test_source_code_drift_cannot_be_promoted(tmp_path):
    path, _, endpoint, evidence = preparation(tmp_path)
    evidence["preflight"]["implementation_sha256"]["llm_client.py"] = "0" * 64
    assert convert_draft(path, endpoint, evidence)[0] is None


def test_successful_candidate_is_written_without_any_approval_or_client(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("No HTTP"))
    source, _, _, evidence = preparation(tmp_path)
    original = copy.deepcopy(evidence)
    evidence_path = tmp_path / "reviewed.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    output = tmp_path / "readiness"
    report = check_readiness(source, output, env_path=config(tmp_path), evidence_path=evidence_path)
    assert report["candidate_ready"]
    assert not report["paid_authorized"]
    assert (output / "candidate_manifest.json").exists()
    assert json.loads(evidence_path.read_text()) == original


def test_report_cli_is_offline_preserves_source_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("No HTTP"))
    source, _, _, _ = preparation(tmp_path)
    before = source.read_bytes()
    output = tmp_path / "readiness"
    report = check_readiness(source, output, env_path=config(tmp_path))
    assert not report["candidate_ready"]
    assert not (output / "candidate_manifest.json").exists()
    assert source.read_bytes() == before
    serialized = (output / "readiness_report.json").read_text()
    assert "secret" not in serialized
    assert (output / "README.md").exists()
    with pytest.raises(FileExistsError):
        check_readiness(source, output, env_path=tmp_path / "pro.env")

import json

import pytest

from experiments.audit_public_interfaces import (
    evaluate_interface, evidence_layers, public_interfaces, verified_snapshots,
)


def test_public_signature_check_does_not_invoke_candidate_function():
    declarations = public_interfaces("def f(x):\n    '''public'''")
    result = evaluate_interface("def f(x): raise AssertionError('must not run')", declarations)
    assert result["status"] == "public_interface_compatible"


def test_missing_public_helper_is_distinct_from_target_signature():
    declarations = public_interfaces("def helper(x):\n    return x\n\ndef f(x):\n    '''public'''")
    result = evaluate_interface("def f(x): return x", declarations)
    assert result["status"] == "public_symbol_missing"
    assert [row["status"] for row in result["symbols"]] == ["missing_or_noncallable", "compatible"]


def test_keyword_names_are_part_of_public_signature_diagnostic():
    declarations = public_interfaces("def f(value):\n    '''public'''")
    result = evaluate_interface("def f(other): return other", declarations)
    assert result["status"] == "public_signature_incompatible"


def test_positional_only_defaults_and_required_keywords_are_supported():
    source = "def f(x, /, y=2, *, z):\n    '''public'''"
    assert evaluate_interface(source, public_interfaces(source))["status"] == "public_interface_compatible"


def test_module_load_error_is_not_misreported_as_missing_function():
    result = evaluate_interface("raise ValueError()", public_interfaces("def f(x): pass"))
    assert result["status"] == "candidate_load_or_process_error"


def test_unparsable_prompt_abstains():
    assert public_interfaces("def f(: bad") == []
    assert evaluate_interface("def f(): pass", [])["status"] == "unsupported_public_prompt"


def test_absent_semantic_coverage_does_not_imply_success():
    assert evidence_layers("unknown", {"public_semantics": []}) == []


def test_semantic_execution_error_is_distinct_from_violation():
    snapshots = {
        "public_inverse": [{
            "request_id": "one", "all_roundtrips_satisfied": False,
            "property_violations": 0, "probe_execution_errors": 1,
        }],
    }
    assert evidence_layers("one", snapshots)[0]["status"] == "check_execution_error"
    snapshots["public_inverse"][0]["property_violations"] = 1
    assert evidence_layers("one", snapshots)[0]["status"] == "property_violation"


def test_stale_semantic_snapshot_refuses_join(tmp_path):
    (tmp_path / "public_semantics_audit.json").write_text(json.dumps({
        "manifest_sha256": "old", "request_content_sha256": "requests", "outputs": [],
    }))
    with pytest.raises(ValueError, match="原始请求不一致"):
        verified_snapshots(tmp_path, "new", "requests")


def test_complete_audit_is_cached_and_keeps_discussion_unevaluated(tmp_path, monkeypatch):
    from experiments import audit_public_interfaces as module
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset": "humaneval", "policies": ["full"],
        "tasks": [{"task_id": "renamed", "problem": "def f(x): pass"}],
    }))
    (tmp_path / "request_one.json").write_text(json.dumps({
        "request_id": "one", "task_id": "renamed", "policy": "full",
        "repeat": 0, "output": "def f(x): return x", "passed": True,
    }))
    monkeypatch.setattr(module, "analyze", lambda path: None)
    calls = []

    def evaluate(output, declarations):
        calls.append(output)
        return {"status": "public_interface_compatible", "symbols": []}

    monkeypatch.setattr(module, "evaluate_interface", evaluate)
    first = module.audit(tmp_path)
    assert module.audit(tmp_path) == first
    assert len(calls) == 1
    assert first["outputs"][0]["discussion_decision"] == "not_evaluated"
    assert first["outputs"][0]["semantic_layers"] == []
    assert first["api_requests"] == 0

import json
import random

import pytest

from experiments.audit_semantic_probes import (
    FAMILIES, coefficients_from_roots, exact_polynomial, evaluate_family, generate_probes,
)
from experiments.public_semantics import ROOT_SUPPORT


def test_generated_probes_have_exact_witnesses_and_valid_domain():
    probes = generate_probes()
    assert len(probes) == 128
    assert len({tuple(probe["coefficients"]) for probe in probes}) == 128
    for probe in probes:
        assert len(probe["coefficients"]) % 2 == 0
        assert probe["coefficients"][-1] != 0
        assert exact_polynomial(probe["coefficients"], probe["witness_root"]) == 0


def test_family_splits_are_disjoint_and_balanced():
    probes = generate_probes()
    development = {tuple(row["coefficients"]) for row in probes if row["split"] == "development"}
    validation = {tuple(row["coefficients"]) for row in probes if row["split"] == "validation"}
    assert len(development) == len(validation) == 64
    assert not development & validation
    assert all(sum(row["family"] == family for row in probes) == 32 for family in FAMILIES)


def test_generation_is_deterministic_and_preserves_global_rng():
    state = random.getstate()
    try:
        assert generate_probes() == generate_probes()
        assert state == random.getstate()
    finally:
        random.setstate(state)


def test_root_expansion_uses_ascending_coefficient_order():
    assert coefficients_from_roots([1, 2, 3]) == [-6, 11, -6, 1]


def test_any_root_is_accepted_without_using_witness_as_expected():
    probes = [{
        "probe_id": "custom", "coefficients": [-6, 11, -6, 1], "witness_root": 1,
    }]
    result = evaluate_family("def solve(xs): return 3", "solve",
                             {"support_code": ROOT_SUPPORT}, probes)
    assert result["all_probes_satisfied"]


def test_invalid_root_and_exception_are_distinguished():
    probes = [{"probe_id": "custom", "coefficients": [-1, 1]}]
    contract = {"support_code": ROOT_SUPPORT}
    wrong = evaluate_family("def solve(xs): return 0", "solve", contract, probes)
    error = evaluate_family("def solve(xs): raise ValueError()", "solve", contract, probes)
    assert wrong["property_violations"] == 1
    assert wrong["check_execution_errors"] == 0
    assert error["property_violations"] == 0
    assert error["check_execution_errors"] == 1


def fake_source(tmp_path):
    problem = (
        "import math\n"
        "def poly(xs,x):\n"
        "    return sum([coeff * math.pow(x,i) for i,coeff in enumerate(xs)])\n"
        "def solve(xs):\n"
        '    """find x such that poly(x) = 0.\n'
        "    >>> solve([-1,1])\n"
        '    1\n    """\n'
    )
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset": "humaneval", "policies": ["full"],
        "tasks": [{"task_id": "renamed", "problem": problem}],
    }))
    (tmp_path / "request_one.json").write_text(json.dumps({
        "task_id": "renamed", "request_id": "one", "policy": "full",
        "repeat": 0, "output": "def solve(xs): return 1",
    }))


def test_probe_runner_freezes_before_execution_and_resumes(tmp_path, monkeypatch):
    from experiments import audit_semantic_probes as module
    fake_source(tmp_path)
    monkeypatch.setattr(module, "analyze", lambda path: None)
    calls = []

    def evaluate(*args):
        assert (tmp_path / "public_probe_manifest.json").exists()
        calls.append(args)
        return {
            "evaluation": {"status": "executed"}, "satisfied": 32,
            "property_violations": 0, "check_execution_errors": 0,
            "all_probes_satisfied": True,
        }

    monkeypatch.setattr(module, "evaluate_family", evaluate)
    module.run(tmp_path, dry_run=True)
    assert not calls
    module.run(tmp_path)
    module.run(tmp_path)
    assert len(calls) == 4
    report = json.loads((tmp_path / "public_probe_audit.json").read_text())
    assert report["complete"]
    assert report["applicable_tasks"] == 1
    assert report["api_requests"] == 0


def test_source_drift_refuses_to_overwrite_frozen_protocol(tmp_path, monkeypatch):
    from experiments import audit_semantic_probes as module
    fake_source(tmp_path)
    monkeypatch.setattr(module, "analyze", lambda path: None)
    module.run(tmp_path, dry_run=True)
    (tmp_path / "request_one.json").write_text("{}")
    with pytest.raises(ValueError, match="协议变更"):
        module.run(tmp_path, dry_run=True)

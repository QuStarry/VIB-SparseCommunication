"""Offline, post-hoc public-evidence audit; not a learned or deployable selector."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_frozen_replay import analyze
from experiments.run_frozen_replay import fingerprint, strict_scalar_equal, write_json


def public_view(task):
    """Only expose problem and candidate messages, never benchmark evaluation material."""
    return {key: task[key] for key in ("task_id", "problem", "candidates")}


def public_spec(task_id):
    """Hand-authored discovery-case evidence; no general task-to-test synthesis claim."""
    if task_id == "HumanEval/93":
        return {
            "source": "literal problem example",
            "checks": [{"expression": "encode('test')", "expected": "TGST"}],
            "interpretation": "Vowel replacement must preserve the swapped case.",
        }
    if task_id == "HumanEval/147":
        return {
            "source": "literal n=5 example and direct enumeration at n=3",
            "checks": [
                {"expression": "get_max_triples(5)", "expected": 1},
                {"expression": "get_max_triples(3)", "expected": 0},
            ],
            "interpretation": "Count residues of i*i-i+1, not residues of i.",
        }
    if task_id == "HumanEval/116":
        return {
            "source": "non-negative, unbounded-integer rule in the problem",
            "checks": [{"expression": "sort_array([1, 4294967296])",
                        "expected": [1, 4294967296]}],
            "specification_conflicts": [
                {"public_input": [1, 5, 2, 3, 4],
                 "literal_example": [1, 2, 3, 4, 5],
                 "rule_derived": [1, 2, 4, 3, 5]},
                {"description": "Non-negative domain is contradicted by a negative example; "
                                "negative encoding width is unspecified."},
            ],
            "interpretation": "32-bit masking fails the non-negative integer rule. "
                              "This does not settle negative-number benchmark semantics "
                              "or explain the archived failure by itself.",
        }
    if task_id == "MATH/1791":
        minimum = 2 * 1**2 + 12 * 1 + 2
        witness = {"a": 1, "b": 3, "c": 9, "d": 3}
        difference = abs(witness["a"] - witness["b"])
        return {
            "source": "Jacobian area derivation and integer lower bound from the problem",
            "derived_minimum": minimum,
            "witness": witness,
            "verified_areas": [
                str(Fraction((witness["c"] - witness["d"])**2, difference)),
                str(Fraction((witness["c"] + witness["d"])**2, difference)),
            ],
            "proof": [
                "|det d(u,v)/d(x,y)|=|a-b|, so area=(intercept difference)^2/|a-b|.",
                "Area ratio 72/18=4 implies max(c,d)=3*min(c,d).",
                "Let min(c,d)=m; 4*m^2=18*|a-b| implies m=3*k, |a-b|=2*k^2.",
                "Positive integer a,b imply a+b>=2*k^2+2; c+d=12*k.",
                "For k>=1, 2*k^2+12*k+2 is increasing; minimum is 16 at k=1.",
            ],
            "interpretation": "Correct continuation coexists with incorrect critiques; "
                              "speaker role or agreement is not an evidence certificate.",
        }
    return {"source": "unsupported", "interpretation": "No public certificate authored."}


EXECUTION_DRIVER = r"""
import json, sys
payload = json.loads(sys.stdin.read())
code = payload["output"].strip()
if code.startswith("```"):
    code = code.split("\n", 1)[1].rsplit("```", 1)[0].strip()
namespace = {}
try:
    exec(code, namespace)
    results = []
    for check in payload["checks"]:
        try:
            actual = eval(check["expression"], namespace)
            results.append({"actual": actual, "satisfied": actual == check["expected"]})
        except Exception as error:
            results.append({"error_type": type(error).__name__, "satisfied": False})
    print(json.dumps({"status": "executed", "checks": results}))
except Exception as error:
    print(json.dumps({"status": "execution_error", "error_type": type(error).__name__}))
"""


def execute_public_checks(output, checks):
    """Run archived local code in a timed child. Process isolation is NOT a sandbox."""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", EXECUTION_DRIVER],
            input=json.dumps({"output": output, "checks": checks}),
            text=True, capture_output=True, timeout=5, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    if result.returncode:
        return {"status": "process_error", "returncode": result.returncode}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"status": "unparseable_process_output"}


def audit_case(task, records):
    """Evaluate discovery outputs against authored public certificates only."""
    visible = public_view(task)
    certificate = public_spec(visible["task_id"])
    outputs = []
    for record in sorted(records, key=lambda row: (row["policy"], row["repeat"])):
        outcome = {
            "policy": record["policy"], "repeat": record["repeat"],
            "request_id": record["request_id"],
            "output_sha256": hashlib.sha256(record["output"].encode()).hexdigest(),
        }
        if "checks" in certificate:
            outcome["public_evaluation"] = execute_public_checks(
                record["output"], certificate["checks"]
            )
        elif "derived_minimum" in certificate:
            outcome["public_evaluation"] = {
                "status": "derived_scalar_comparison",
                "satisfied": strict_scalar_equal(
                    record["output"], str(certificate["derived_minimum"])
                ),
            }
        else:
            outcome["public_evaluation"] = {"status": "unsupported"}
        outputs.append(outcome)
    return {
        **visible, "certificate": certificate, "outputs": outputs,
        "certificate_authoring": "manual_post_hoc_discovery_only",
        "candidate_message_attribution": "not_intervened_not_proven",
    }


def audit(directory):
    boundary = analyze(directory)  # Labels select discovery cases, never author evidence.
    contrast = boundary["collaboration_boundary"]["contrasts"]["full_minus_first_round"]
    selected_ids = {
        row["task_id"] for row in contrast["tasks"]
        if row["observed_direction"] != "tie" or row["opposite_repeat_differences"]
    }
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = sorted(directory.glob("request_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    cases = [
        audit_case(
            public_view(task),
            [{key: row[key] for key in ("request_id", "policy", "repeat", "output")}
             for row in records if row["task_id"] == task["task_id"]
             and row["policy"] in ("full", "first_round", "first_round_control")],
        )
        for task in manifest["tasks"] if task["task_id"] in selected_ids
    ]
    return {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "request_content_sha256": fingerprint([
            [path.name, hashlib.sha256(path.read_bytes()).hexdigest()] for path in paths
        ]),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coverage": {
            "contrast": "full_minus_first_round",
            "selected_cases": len(selected_ids), "audited_cases": len(cases),
            "unsupported_cases": sum(case["certificate"]["source"] == "unsupported"
                                     for case in cases),
        },
        "api_requests": 0,
        "cases": cases,
        "limitations": [
            "Selection uses observed pass differences; evidence generation never reads "
            "reference answers or hidden tests. This is discovery, not blinded validation.",
            "Public certificates are authored manually after seeing outcomes; "
            "no automatic verifier or policy performance is established.",
            "No message-removal or replacement intervention was run; "
            "compatibility with a bad suggestion does not prove its causal effect.",
            "Only observed Full/first-round discordances are covered; "
            "this is not an all-task false-positive or recall evaluation.",
            "Child processes have timeouts but no filesystem/network security sandbox; "
            "only previously archived local outputs are executed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    for directory in args.directories:
        report = audit(directory)
        path = directory / "public_evidence_audit.json"
        write_json(path, report)
        print(json.dumps({"directory": str(directory), **report["coverage"]}))


if __name__ == "__main__":
    main()

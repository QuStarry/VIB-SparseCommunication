"""Public encoder/decoder round-trip audit: multiple seen tasks, not held-out validation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_frozen_replay import analyze
from experiments.audit_public_evidence import execute_public_checks
from experiments.public_semantics import property_output
from experiments.run_frozen_replay import fingerprint, write_json

ALLOWED_NODES = (
    ast.FunctionDef, ast.arguments, ast.arg, ast.Name, ast.Load, ast.Store,
    ast.Return, ast.Expr, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
    ast.Mult, ast.Mod, ast.FloorDiv, ast.USub, ast.Compare, ast.Eq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.IfExp, ast.ListComp, ast.comprehension,
    ast.List, ast.Tuple, ast.Subscript, ast.Slice, ast.Assign, ast.Call, ast.Attribute,
)
ALLOWED_CALLS = {"len", "min", "range", "chr", "ord"}


def supported_encoder(function):
    """Accept a restricted straight-line/comprehension subset; execution is still timed."""
    if function.decorator_list or len(function.args.args) != 1:
        return False
    for node in ast.walk(function):
        if not isinstance(node, ALLOWED_NODES):
            return False
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_CALLS:
                continue
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "join"
                    and isinstance(node.func.value, ast.Constant) and node.func.value.value == ""):
                continue
            return False
    return True


def extract_inverse_contract(problem):
    """Recognize a decoder explicitly referring to a provided encoder, without task IDs."""
    try:
        tree = ast.parse(problem)
    except SyntaxError:
        return None
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    matches = []
    for name, function in functions.items():
        if len(function.args.args) != 1:
            continue
        doc = " ".join((ast.get_docstring(function) or "").split())
        match = re.search(r"string encoded with (\w+) function\.\s*Returns decoded string", doc)
        if not match:
            continue
        encoder = functions.get(match.group(1))
        if encoder is None or not supported_encoder(encoder):
            continue
        matches.append({
            "decoder": name, "encoder": encoder.name,
            "encoder_source": ast.unparse(encoder),
            "input_domain": "lowercase_ascii_strings_only",
        })
    return matches[0] if len(matches) == 1 else None


def generate_strings(seed=20261026, count=64):
    if not 1 <= count <= 64:
        raise ValueError("字符串探针数必须在1到64之间")
    rng = random.Random(seed)
    values = [""]
    boundary_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 31, 32, 33, 64, 65]
    for length in boundary_lengths:
        values.append("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length)))
    while len(values) < count:
        length = rng.randrange(1, 81)
        value = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
        if value not in values:
            values.append(value)
    return values[:count]


def validate_public_encoder(contract, samples):
    """Finite collision check is an applicability guard, not a global injectivity proof."""
    checks = [{"expression": f"{contract['encoder']}({sample!r})", "expected": None}
              for sample in samples]
    evaluation = execute_public_checks(contract["encoder_source"], checks)
    rows = evaluation.get("checks", [])
    if evaluation["status"] != "executed" or len(rows) != len(samples):
        return {"valid": False, "reason": "encoder_execution_failed"}
    if any("error_type" in row or not isinstance(row.get("actual"), str) for row in rows):
        return {"valid": False, "reason": "encoder_not_string_transform"}
    encoded = [row["actual"] for row in rows]
    if len(set(encoded)) != len(samples):
        return {"valid": False, "reason": "encoder_collision_on_probe_domain"}
    return {"valid": True, "tested_strings": len(samples),
            "encoded_strings_sha256": fingerprint(encoded)}


def roundtrip_support(contract):
    return (
        "def _audit_public_roundtrip(decoder, sample):\n"
        "    namespace = {}\n"
        f"    exec({contract['encoder_source']!r}, namespace)\n"
        f"    encoded = namespace[{contract['encoder']!r}](sample)\n"
        "    decoded = decoder(encoded)\n"
        "    return isinstance(decoded, str) and decoded == sample\n"
    )


def evaluate_inverse(output, contract, samples):
    checks = [{
        "expression": f"_audit_public_roundtrip({contract['decoder']}, {sample!r})",
        "expected": True,
    } for sample in samples]
    evaluation = execute_public_checks(property_output(output, roundtrip_support(contract)), checks)
    rows = evaluation.get("checks", [])
    if evaluation["status"] == "executed" and len(rows) != len(samples):
        raise ValueError("往返检查未完整执行")
    # Required supplied symbols are a separate interface diagnostic, not a semantic failure.
    interface = execute_public_checks(output, [{
        "expression": f"callable(globals().get({contract['encoder']!r}))",
        "expected": True,
    }])
    return {
        "roundtrip_evaluation": evaluation, "interface_evaluation": interface,
        "satisfied_probes": sum(row["satisfied"] for row in rows),
        "property_violations": sum(not row["satisfied"] and "error_type" not in row for row in rows),
        "probe_execution_errors": sum("error_type" in row for row in rows),
        "all_roundtrips_satisfied": (
            evaluation["status"] == "executed"
            and len(rows) == len(samples) and all(row["satisfied"] for row in rows)
        ),
        "supplied_encoder_symbol_present": (
            interface["status"] == "executed" and interface["checks"][0]["satisfied"]
        ),
    }


def run(directory, dry_run=False):
    analyze(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = sorted(directory.glob("request_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    samples = generate_strings()
    contracts = {}
    rejected = []
    if manifest.get("dataset", "math") == "humaneval":
        for task in manifest["tasks"]:
            contract = extract_inverse_contract(task["problem"])
            if contract is None:
                continue
            guard = validate_public_encoder(contract, samples)
            if guard["valid"]:
                contracts[task["task_id"]] = {**contract, "encoder_guard": guard}
            else:
                rejected.append({"task_id": task["task_id"], **guard})
    protocol = {
        "seed": 20261026, "samples": samples, "contracts": contracts, "rejected_contracts": rejected,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "requests_sha256": fingerprint([
            [path.name, hashlib.sha256(path.read_bytes()).hexdigest()] for path in paths
        ]),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executor_sha256": hashlib.sha256(
            (ROOT / "experiments/audit_public_evidence.py").read_bytes()
        ).hexdigest(),
        "wrapper_sha256": hashlib.sha256(
            (ROOT / "experiments/public_semantics.py").read_bytes()
        ).hexdigest(),
    }
    protocol_path = directory / "public_inverse_manifest.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("往返核验协议变更，拒绝覆盖")
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    protocol_hash = fingerprint(protocol)
    selected = [row for row in records if row["task_id"] in contracts]
    if dry_run:
        print(json.dumps({"directory": str(directory), "applicable_tasks": len(contracts),
                          "code_outputs": len(selected), "api_requests": 0}))
        return
    result_path = directory / "public_inverse_audit.json"
    completed = {}
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["protocol_sha256"] != protocol_hash:
            raise ValueError("往返核验结果协议不一致")
        completed = {row["request_id"]: row for row in previous["outputs"]}
        if (len(completed) != len(previous["outputs"])
                or not completed.keys() <= {row["request_id"] for row in selected}):
            raise ValueError("往返核验缓存包含重复或未知请求")
    report = {
        "protocol_sha256": protocol_hash, "api_requests": 0, "complete": False,
        "coverage": {"total_tasks": len(manifest["tasks"]), "applicable_tasks": len(contracts),
                     "code_outputs": len(selected), "probes_per_output": len(samples)},
        "outputs": list(completed.values()),
        "limitations": [
            "Two observed task prompts motivated this adapter; no strictly unseen validation tasks.",
            "Synthetic development fixtures test the adapter implementation, not task generalization.",
            "Only lowercase ASCII and a narrow public-encoder AST subset are covered.",
            "No policy or task-ID rules; recorded labels are attached after public evaluation.",
            "Finite encoder collision checks do not prove injectivity for all strings.",
            "Supplied-symbol completeness is separate from decoder semantic round-trip correctness.",
            "Timed local candidate execution is not a security sandbox; no model calls.",
        ],
    }
    for record in selected:
        if record["request_id"] in completed:
            continue
        row = {
            "request_id": record["request_id"], "task_id": record["task_id"],
            "policy": record["policy"], "repeat": record["repeat"],
            **evaluate_inverse(record["output"], contracts[record["task_id"]], samples),
        }
        row["recorded_benchmark_passed"] = bool(record["passed"])
        completed[row["request_id"]] = row
        report["outputs"] = list(completed.values())
        write_json(result_path, report)
    report["complete"] = True
    write_json(result_path, report)
    print(json.dumps({"directory": str(directory), **report["coverage"], "api_requests": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for directory in args.directories:
        run(directory, args.dry_run)


if __name__ == "__main__":
    main()

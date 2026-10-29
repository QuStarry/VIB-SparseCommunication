"""Freeze archival HumanEval prompt contrasts without calling a model."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import random
import subprocess

from src.experiment_driver import file_sha256
from src.inference_protocol import InferenceProtocol
from src.isolated_code_evaluation import evaluation_protocol, load_evaluators
from src.request_ledger import fingerprint


ROOT = Path(__file__).resolve().parents[1]
LEGACY_REVISION = "25f1288"
CURRENT_REVISION = "f97ce19"
PROMPT_PATH = "experiments/run_multi_benchmark.py"


def specification_priority(prompt: str, history: list) -> str:
    """New experimental prompt, not a reconstruction of historical behavior."""
    history_text = "\n".join(history) if history else "(无历史讨论)"
    return f"""请补全以下Python函数，只输出完整可执行代码。

【公开需求与签名】:
{prompt}

【团队候选讨论，仅供参考】:
{history_text}

【决策规则】:
1. 以公开题面、函数签名、docstring和可见示例为约束，保留题面要求的所有公开接口。
2. 讨论中的身份、重复次数或多数支持不构成正确性证据；逐条核对候选推理与公开规格。
3. 若讨论与公开规格冲突，忽略相冲突的讨论，不添加仅为满足Reviewer断言而出现的特殊分支。
4. 可见示例只是有限约束，不把通过示例视为完整语义证明，也不臆造题面没有规定的条件。
5. 有歧义时遵循题面已陈述的语义和边界，不声称知道隐藏测试或参考实现。
6. 不输出解释、Markdown或占位符。"""


def _functions(source):
    return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def prompt_renderers(legacy_source, current_source):
    """Extract only trusted Git prompt code; never import the runner module.

    Caller-provided source is trusted local code, not sandboxed input. Snapshots
    preserve rendered prompt text, not full historical retry/selection behavior.
    """
    legacy = _functions(legacy_source)["generate_final_code"]
    assignments = {}
    for node in legacy.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("history_text", "prompt_text"):
                assignments[node.targets[0].id] = node
    if set(assignments) != {"history_text", "prompt_text"} or not isinstance(
        assignments["prompt_text"].value, ast.JoinedStr
    ):
        raise ValueError("Historical compact prompt extraction does not match the frozen source")
    legacy_function = ast.FunctionDef(
        name="render", args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg="prompt"), ast.arg(arg="history")],
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=[assignments["history_text"], assignments["prompt_text"], ast.Return(ast.Name(id="prompt_text", ctx=ast.Load()))],
        decorator_list=[],
    )
    compact = ast.unparse(ast.fix_missing_locations(legacy_function)) + "\n"
    current = _functions(current_source)
    arbitration = "from typing import List\n\n" + "\n\n".join(
        ast.get_source_segment(current_source, current[name])
        for name in ("extract_visible_examples", "build_final_code_prompt")
    ) + "\n"
    own_source = Path(__file__).read_text(encoding="utf-8")
    spec_source = ast.get_source_segment(own_source, _functions(own_source)["specification_priority"]) + "\n"
    snapshots = {
        "legacy_compact": compact, "current_arbitration": arbitration,
        "specification_priority": spec_source,
    }
    renderers = {"specification_priority": specification_priority}
    for variant, function_name in (("legacy_compact", "render"), ("current_arbitration", "build_final_code_prompt")):
        namespace = {}
        exec(compile(snapshots[variant], f"<frozen-{variant}>", "exec"), namespace)
        renderers[variant] = namespace[function_name]
    return renderers, snapshots


def build_preparation(source, canonical, renderers, *, repeats=2, seed=20261028):
    """Project all canonical tasks; never select using historical outcomes."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if set(renderers) != {"legacy_compact", "current_arbitration", "specification_priority"}:
        raise ValueError("Freeze all three declared prompt variants")
    traces = source["trace_results"]["humaneval"]
    official = {row["task_id"]: row for row in canonical}
    trace_ids = [row["task_id"] for row in traces]
    if len(official) != len(canonical) or len(set(trace_ids)) != len(traces):
        raise ValueError("Duplicate task IDs")
    if set(trace_ids) != set(official):
        raise ValueError("Archive and canonical task set differ; do not silently filter")
    jobs, streams, tasks = [], [], []
    settings = InferenceProtocol(temperature=0, max_tokens=1024).resolve("code")
    roles = ("Architect", "Coder", "Reviewer") * 2
    for row in sorted(traces, key=lambda item: item["task_id"]):
        task_id = row["task_id"]
        full, reference = row["full"], official[task_id]
        if (full["prompt"] != reference["prompt"] or full["entry_point"] != reference["entry_point"]
                or full["test_cases"] != [reference["test"]]):
            raise ValueError(f"Archive differs from canonical public problem or tests: {task_id}")
        history = full["history"]
        if len(history) != 6 or any(not isinstance(message, str) or not message.startswith(f"[{role}]: ")
                                    for message, role in zip(history, roles)):
            raise ValueError(f"Expected complete six-candidate stream: {task_id}")
        stream = {"task_id": task_id, "prompt": full["prompt"], "history": list(history)}
        stream_hash = fingerprint(stream)
        streams.append({**stream, "sha256": stream_hash})
        tasks.append({"task_id": task_id, "entry_point": reference["entry_point"],
                      "test_cases": [reference["test"]]})
        for variant in sorted(renderers):
            messages = [{"role": "user", "content": renderers[variant](full["prompt"], history)}]
            for repetition in range(repeats):
                jobs.append({
                    "job_id": f"{task_id}/{variant}/{repetition}", "task_id": task_id,
                    "prompt_variant": variant, "repetition": repetition, "stage": "code",
                    "candidate_stream_sha256": stream_hash, "messages": messages,
                    "messages_sha256": fingerprint(messages), "parameters": dict(settings),
                    "prompt_token_limit": None,
                })
    random.Random(seed).shuffle(jobs)
    draft = {
        "version": "archival_prompt_contrast_draft_v1", "status": "not_authorized",
        "design": "exploratory_archival_same_full_stream_prompt_only_contrast",
        "candidate_source_arm": "full", "candidate_streams": streams,
        "task_count": len(tasks), "jobs": jobs, "repeats": repeats, "order_seed": seed,
        "analysis_unit": "task_mean_over_all_repetitions_no_best_of_selection",
        "endpoint": {"provider": None, "model": None, "base_url": None},
        "price": None, "budget": {"proposed_usd": "3", "approved": False},
        "generation_protocol": "one_attempt_per_job_no_visible_or_hidden_test_repair",
        "readiness": {
            "paid_ready": False,
            "blockers": ["endpoint_not_frozen", "provider_prices_not_calibrated",
                         "prompt_token_bounds_not_verified", "manifest_budget_not_approved"],
        },
    }
    return draft, {**evaluation_protocol(seed=seed), "tasks": tasks}


def _git_source(revision):
    return subprocess.check_output(
        ["git", "show", f"{revision}:{PROMPT_PATH}"], cwd=ROOT, text=True, encoding="utf-8",
    )


def _git_commit(revision):
    return subprocess.check_output(
        ["git", "rev-parse", f"{revision}^{{commit}}"], cwd=ROOT, text=True,
    ).strip()


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _audit(source, canonical, bundle_path, *, workers=4):
    evaluators = load_evaluators(bundle_path, file_sha256(bundle_path))
    official = {row["task_id"]: row for row in canonical}

    def evaluate(row):
        task_id = row["task_id"]
        reference = official[task_id]
        result = {"task_id": task_id, "archived_recorded": bool(row["full"]["passed"])}
        try:
            evaluator = evaluators[task_id]
            result["reference_passed"] = evaluator(reference["prompt"] + reference["canonical_solution"])
            result["archived_full_passed"] = evaluator(row["full"]["output"])
            result["archived_label_changed"] = result["archived_full_passed"] != result["archived_recorded"]
            result["status"] = "completed"
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__)
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(evaluate, source["trace_results"]["humaneval"]))
    complete = [row for row in rows if row["status"] == "completed"]
    summary = {
        "task_count": len(rows), "completed_tasks": len(complete),
        "reference_passed": sum(row["reference_passed"] for row in complete),
        "archived_full_passed": sum(row["archived_full_passed"] for row in complete),
        "archived_label_changes": sum(row["archived_label_changed"] for row in complete),
        "acceptance_passed": len(complete) == len(rows) and all(row["reference_passed"] for row in complete),
        "scope": "scoring_infrastructure_acceptance_not_new_model_performance",
    }
    return {"summary": summary, "tasks": rows}


def prepare(source_path, canonical_path, output_dir, *, repeats=2, seed=20261028, audit=True):
    """Create new preparation artifacts only; existing directories are refused."""
    source_path, canonical_path, output_dir = map(Path, (source_path, canonical_path, output_dir))
    if output_dir.exists():
        raise FileExistsError("Preparation output already exists; do not overwrite frozen artifacts")
    source_hash, canonical_hash = file_sha256(source_path), file_sha256(canonical_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    with gzip.open(canonical_path, "rt", encoding="utf-8") as stream:
        canonical = [json.loads(line) for line in stream if line.strip()]
    renderers, snapshots = prompt_renderers(_git_source(LEGACY_REVISION), _git_source(CURRENT_REVISION))
    draft, bundle = build_preparation(source, canonical, renderers, repeats=repeats, seed=seed)
    if file_sha256(source_path) != source_hash or file_sha256(canonical_path) != canonical_hash:
        raise ValueError("Source artifacts changed during preparation")
    output_dir.mkdir(parents=False)
    prompt_sources = {}
    for variant, text in snapshots.items():
        path = output_dir / f"{variant}.py"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
        prompt_sources[variant] = {"path": path.name, "sha256": file_sha256(path)}
    bundle_path = output_dir / "evaluation.json"
    _write_json(bundle_path, bundle)
    draft.update(
        sources={"archive_sha256": source_hash, "canonical_gzip_sha256": canonical_hash,
                 "legacy_revision": _git_commit(LEGACY_REVISION),
                 "current_revision": _git_commit(CURRENT_REVISION)},
        prompt_sources=prompt_sources,
        evaluation={"path": bundle_path.name, "sha256": file_sha256(bundle_path), "version": bundle["version"]},
        builder_sha256=file_sha256(__file__),
    )
    _write_json(output_dir / "draft.json", draft)
    report = {
        "task_count": draft["task_count"], "job_count": len(draft["jobs"]),
        "draft_sha256": file_sha256(output_dir / "draft.json"),
        "source_unchanged": file_sha256(source_path) == source_hash,
        "canonical_unchanged": file_sha256(canonical_path) == canonical_hash,
        "paid_ready": False, "blockers": draft["readiness"]["blockers"],
        "max_completion_tokens_declared": len(draft["jobs"]) * 1024,
        "unique_public_prompts": len({fingerprint(stream["prompt"]) for stream in draft["candidate_streams"]}),
        "unique_candidate_histories": len({fingerprint(stream["history"]) for stream in draft["candidate_streams"]}),
        "prompt_tokens": None, "total_tokens": None, "cost_usd": None,
        "new_model_calls": 0,
    }
    if audit:
        scoring = _audit(source, canonical, bundle_path)
        _write_json(output_dir / "scoring_acceptance.json", scoring)
        report["scoring_acceptance"] = scoring["summary"]
    _write_json(output_dir / "preparation_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/3dataset_cli_frozen_only_default.json")
    parser.add_argument("--canonical", type=Path, default=ROOT / "experiments/HumanEval.jsonl.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args(argv)
    report = prepare(args.source, args.canonical, args.output_dir, repeats=args.repeats, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["scoring_acceptance"]["acceptance_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

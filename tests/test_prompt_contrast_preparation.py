import copy
import json
from pathlib import Path
import socket
import subprocess

import pytest

from experiments.prepare_prompt_contrast import build_preparation, prompt_renderers, prepare


ROOT = Path(__file__).resolve().parents[1]


def renderers():
    legacy = subprocess.check_output(
        ["git", "show", "25f1288:experiments/run_multi_benchmark.py"], cwd=ROOT, text=True,
    )
    current = subprocess.check_output(
        ["git", "show", "f97ce19:experiments/run_multi_benchmark.py"], cwd=ROOT, text=True,
    )
    return prompt_renderers(legacy, current)[0]


def fixture():
    problem = 'def f(x):\n    """Return x+1."""\n'
    test = "def check(candidate):\n    assert candidate(1) == 2"
    full = {"prompt": problem, "history": [f"[{role}]: candidate {index}"
            for index, role in enumerate(["Architect", "Coder", "Reviewer"] * 2)],
            "test_cases": [test], "entry_point": "f", "passed": True,
            "output": "old-output", "reference_answer": "SECRET_REFERENCE"}
    source = {"trace_results": {"humaneval": [{"task_id": "HumanEval/0", "full": full}]}}
    canonical = [{"task_id": "HumanEval/0", "prompt": problem, "test": test, "entry_point": "f",
                  "canonical_solution": "    return x+1"}]
    return source, canonical


def test_all_prompt_variants_share_frozen_history_and_parameters():
    source, canonical = fixture()
    draft, bundle = build_preparation(source, canonical, renderers(), repeats=2, seed=7)
    assert len(draft["jobs"]) == 6
    assert len({job["candidate_stream_sha256"] for job in draft["jobs"]}) == 1
    assert {job["prompt_variant"] for job in draft["jobs"]} == {
        "legacy_compact", "current_arbitration", "specification_priority",
    }
    assert all(job["parameters"] == {"temperature": 0, "max_tokens": 1024} for job in draft["jobs"])
    assert draft["readiness"]["paid_ready"] is False
    assert draft["price"] is None
    assert all(job["prompt_token_limit"] is None for job in draft["jobs"])
    assert "SECRET_REFERENCE" not in json.dumps(draft)
    assert "def check(candidate)" not in json.dumps(draft)
    assert len(bundle["tasks"]) == 1


def test_old_labels_outputs_and_references_do_not_change_jobs():
    source, canonical = fixture()
    original = copy.deepcopy(source)
    first, _ = build_preparation(source, canonical, renderers())
    source["trace_results"]["humaneval"][0]["full"].update(
        passed=False, output="changed", reference_answer="changed",
    )
    second, _ = build_preparation(source, canonical, renderers())
    assert first["jobs"] == second["jobs"]
    assert original["trace_results"]["humaneval"][0]["full"]["passed"] is True


@pytest.mark.parametrize("field", ["prompt", "entry_point", "test_cases"])
def test_official_public_problem_or_hidden_test_drift_is_rejected(field):
    source, canonical = fixture()
    source["trace_results"]["humaneval"][0]["full"][field] = ["changed"] if field == "test_cases" else "changed"
    with pytest.raises(ValueError, match="canonical"):
        build_preparation(source, canonical, renderers())


def test_missing_tasks_are_not_silently_dropped():
    source, canonical = fixture()
    canonical.append({**canonical[0], "task_id": "HumanEval/1"})
    with pytest.raises(ValueError, match="task set"):
        build_preparation(source, canonical, renderers())


def test_incomplete_candidate_stream_is_rejected():
    source, canonical = fixture()
    source["trace_results"]["humaneval"][0]["full"]["history"].pop()
    with pytest.raises(ValueError, match="six"):
        build_preparation(source, canonical, renderers())


def test_preparation_does_not_mutate_archival_source():
    source, canonical = fixture()
    before = copy.deepcopy(source)
    build_preparation(source, canonical, renderers())
    assert source == before


def test_extracted_current_prompt_equals_original_function_without_importing_runner():
    legacy = subprocess.check_output(
        ["git", "show", "25f1288:experiments/run_multi_benchmark.py"], cwd=ROOT, text=True,
    )
    current = subprocess.check_output(
        ["git", "show", "f97ce19:experiments/run_multi_benchmark.py"], cwd=ROOT, text=True,
    )
    functions, snapshots = prompt_renderers(legacy, current)
    namespace = {}
    exec(snapshots["current_arbitration"], namespace)
    problem = 'def f(x):\n    """\n    >>> f(1)\n    2\n    """\n'
    assert functions["current_arbitration"](problem, ["candidate"]) == namespace["build_final_code_prompt"](
        problem, ["candidate"],
    )


def test_historical_compact_prompt_is_not_a_newly_invented_baseline():
    text = renderers()["legacy_compact"]("TASK", ["A", "B"])
    assert "1. 仅输出最终可执行的 Python 代码实现。" in text
    assert "Reviewer" not in text
    assert "A\nB" in text
    assert "0. 【第 0 步" not in text


def test_cli_prepares_actual_archival_set_without_network_or_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("No network"))
    source = ROOT / "results/3dataset_cli_frozen_only_default.json"
    canonical = ROOT / "experiments/HumanEval.jsonl.gz"
    output = tmp_path / "preparation"
    report = prepare(source, canonical, output, audit=False)
    assert report["task_count"] == 164
    assert report["job_count"] == 984
    assert not report["paid_ready"]
    draft = json.loads((output / "draft.json").read_text())
    assert len({job["task_id"] for job in draft["jobs"]}) == 164
    with pytest.raises(FileExistsError):
        prepare(source, canonical, output, audit=False)

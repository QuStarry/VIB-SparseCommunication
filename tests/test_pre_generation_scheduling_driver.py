import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

import run_pre_generation_scheduling as driver  # noqa: E402
from run_multi_benchmark import BypassVIBRouter, VIBSparseCommRouter  # noqa: E402


class _FakeHarness:
    """Minimal stand-in: build_condition only needs _create_vib_router."""

    def __init__(self):
        self.salts = []

    def _create_vib_router(self, task_salt=None):
        self.salts.append(task_salt)
        return VIBSparseCommRouter(beta_0=3.2, lambda_decay=0.08, adaptive=True, alpha=0.35, tau_min=-1.0)


def _make_record(task_id, cond_values):
    return {"task_id": task_id, "dataset": "MATH", "conditions": cond_values}


def _cond(passed, messages, candidates, skipped, prompt, completion, total, error=None):
    return {
        "passed": passed, "messages": messages, "candidates": candidates,
        "skipped_roles": skipped, "prompt_tokens": prompt,
        "completion_tokens": completion, "total_tokens": total, "error": error,
    }


def test_collect_task_ids_walks_nested_manifest():
    manifest = {
        "dataset": "HumanEval",
        "tasks": [{"task_id": "HumanEval/1"}, {"task_id": "HumanEval/2"}],
        "results": {"task_ids": ["HumanEval/3"]},
    }
    sink = set()
    driver.collect_task_ids(manifest, sink)
    assert sink == {"HumanEval/1", "HumanEval/2", "HumanEval/3"}


def test_collect_task_ids_handles_plain_list_and_strings():
    sink = set()
    driver.collect_task_ids(["MATH/1", {"task_id": "MATH/2"}], sink)
    assert sink == {"MATH/1", "MATH/2"}


def test_collect_task_ids_walks_detailed_results_style_payload():
    # Mirrors results/gsm8k_main_1319_t0.json: payload lives under
    # detailed_results -> <dataset> -> list of records, which an earlier
    # key-whitelist walk silently skipped.
    manifest = {
        "summary": {"gsm8k": {"total_tasks": 2}},
        "detailed_results": {
            "gsm8k": [{"task_id": "GSM8K/0"}, {"task_id": "GSM8K/1"}],
        },
    }
    sink = set()
    driver.collect_task_ids(manifest, sink)
    assert {"GSM8K/0", "GSM8K/1"}.issubset(sink)


def test_load_exclusions_merges_inline_and_manifest(tmp_path):
    manifest_path = tmp_path / "hist.json"
    manifest_path.write_text(
        json.dumps({"tasks": [{"task_id": "HumanEval/32"}, {"task_id": "HumanEval/7"}]}),
        encoding="utf-8",
    )
    excluded = driver.load_exclusions([str(manifest_path)], ["MATH/1791"])
    assert excluded == {"HumanEval/32", "HumanEval/7", "MATH/1791"}


def test_build_condition_full_bypasses_gate_and_has_no_scheduler():
    router, scheduler = driver.build_condition("full", "HumanEval", _FakeHarness(), "HumanEval/1", 7)
    assert isinstance(router, BypassVIBRouter)
    assert scheduler is None


def test_build_condition_post_gate_keeps_gate_without_scheduler():
    harness = _FakeHarness()
    router, scheduler = driver.build_condition("post_gate", "HumanEval", harness, "HumanEval/1", 7)
    assert isinstance(router, VIBSparseCommRouter)
    assert scheduler is None
    assert harness.salts == ["HumanEval/1"]


def test_build_condition_fixed_trim_runs_only_first_round():
    _, scheduler = driver.build_condition("fixed_trim", "MATH", _FakeHarness(), "MATH/1", 7)
    names = ["Planner", "Solver", "Verifier"]
    assert scheduler("goal", [], 1, names) == names
    assert scheduler("goal", ["[Planner]: x"], 2, names) == []


def test_build_condition_coverage_excludes_represented_roles_and_keeps_closer():
    _, scheduler = driver.build_condition("coverage", "MATH", _FakeHarness(), "MATH/1", 7)
    names = ["Planner", "Solver", "Verifier"]
    selected = scheduler("goal", ["[Planner]: plan"], 2, names)
    assert selected == ["Solver", "Verifier"]


def test_build_condition_random_skip_is_count_matched_and_deterministic():
    _, scheduler = driver.build_condition("random_skip", "MATH", _FakeHarness(), "MATH/1", 7)
    names = ["Planner", "Solver", "Verifier"]
    history = ["[Planner]: plan"]  # coverage keeps 2 of 3 from round 2 on
    first = scheduler("goal", history, 2, names)
    second = scheduler("goal", history, 2, names)
    assert first == second  # deterministic per (task, round)
    assert len(first) == 2  # same count as coverage
    assert set(first).issubset(set(names))
    assert scheduler("goal", [], 1, names) == names  # first round keeps everyone


def test_build_condition_rejects_unknown_name():
    try:
        driver.build_condition("nope", "MATH", _FakeHarness(), "MATH/1", 7)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown condition")


def test_paired_bootstrap_ci_constant_differences():
    ci = driver.paired_bootstrap_ci([2.0, 2.0, 2.0, 2.0], n_boot=200, seed=1)
    assert ci["mean"] == 2.0
    assert ci["lo"] == 2.0 and ci["hi"] == 2.0
    assert ci["n"] == 4


def test_paired_bootstrap_ci_empty_returns_none():
    assert driver.paired_bootstrap_ci([], n_boot=200) is None


def test_summarize_conditions_aggregates_totals():
    records = [
        _make_record("MATH/1", {
            "full": _cond(True, 6, 6, 0, 100, 50, 150),
            "coverage": _cond(True, 4, 4, 2, 80, 40, 120),
        }),
        _make_record("MATH/2", {
            "full": _cond(False, 6, 6, 0, 200, 100, 300),
            "coverage": _cond(False, 5, 5, 1, 150, 60, 210),
        }),
    ]
    summary = driver.summarize_conditions(records, ["full", "coverage"])
    assert summary["full"]["accuracy_pct"] == 50.0
    assert summary["coverage"]["accuracy_pct"] == 50.0
    assert summary["full"]["total_tokens"] == 450
    assert summary["coverage"]["total_tokens"] == 330
    assert summary["coverage"]["skipped_roles"] == 3


def test_compare_conditions_reports_paired_delta_and_reduction():
    records = [
        _make_record("MATH/1", {
            "full": _cond(True, 6, 6, 0, 100, 50, 150),
            "coverage": _cond(True, 4, 4, 2, 80, 40, 120),
        }),
        _make_record("MATH/2", {
            "full": _cond(False, 6, 6, 0, 200, 100, 300),
            "coverage": _cond(True, 5, 5, 1, 150, 60, 210),
        }),
    ]
    summary = driver.summarize_conditions(records, ["full", "coverage"])
    cmp = driver.compare_conditions(records, "coverage", "full", summary, n_boot=200, seed=1)
    assert cmp["n_pairs"] == 2
    # coverage - full per task: MATH/1 (1-1)=0, MATH/2 (1-0)=100 -> mean 50.0
    assert cmp["accuracy_delta_pp"]["mean"] == 50.0
    assert round(cmp["reduction_pct"]["total_tokens"], 2) == 26.67


def test_record_for_condition_reads_workflow_stats_and_token_usage():
    variant = {
        "passed": True,
        "workflow_stats": {
            "total_approved": 5,
            "total_candidates": 6,
            "schedule_decisions": [
                {"selected": ["Solver"], "skipped": ["Planner", "Verifier"]},
            ],
        },
        "token_stats": {"total_prompt_tokens": 90, "total_completion_tokens": 30, "total_tokens": 120},
    }
    record = driver._record_for_condition(variant)
    assert record["passed"] is True
    assert record["messages"] == 5
    assert record["candidates"] == 6
    assert record["skipped_roles"] == 2
    assert record["total_tokens"] == 120
    assert record["error"] is None


def test_record_for_condition_marks_errors_as_zero_cost():
    record = driver._record_for_condition({}, error="RuntimeError: boom")
    assert record["passed"] is False
    assert record["total_tokens"] == 0
    assert record["error"] == "RuntimeError: boom"


def test_parse_args_rejects_missing_output_dir():
    import pytest
    with pytest.raises(SystemExit):
        driver.parse_args([])


def test_task_id_sort_key_orders_numerically_and_tolerates_non_numeric():
    assert driver._task_id_sort_key("MBPP/602") < driver._task_id_sort_key("MBPP/1000")
    assert driver._task_id_sort_key("GSM8K-train/2") < driver._task_id_sort_key("GSM8K-train/10")
    # 非数字后缀不抛异常，退化为字符串序
    assert driver._task_id_sort_key("Weird/abc") == ("Weird", 1, "abc")


def test_select_samples_orders_naturally_so_protocol_range_holds(monkeypatch):
    """取前 N 题须按数值升序，否则 ``GSM8K-train/0..299`` 的协议区间不成立。"""
    from run_multi_benchmark import TaskSample

    class _FakeAdapter:
        def load(self, limit=-1):
            ids = [f"GSM8K-train/{i}" for i in (2, 10, 1, 300, 0)]
            return [TaskSample(task_id=i, dataset_name="GSM8K", prompt="q") for i in ids]

    monkeypatch.setattr(
        driver.DatasetFactory, "get_adapter", staticmethod(lambda name: _FakeAdapter())
    )

    samples = driver._select_samples("gsm8k_train", 3, set())
    assert [s.task_id for s in samples] == ["GSM8K-train/0", "GSM8K-train/1", "GSM8K-train/2"]

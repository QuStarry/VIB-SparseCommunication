import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(module_path.parent))
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_trace_case():
    return {
        "task_id": "GSM8K/227",
        "dataset_name": "GSM8K",
        "full": {
            "prompt": "question",
            "reference_answer": "9",
            "history": ["[Planner]: sum costs", "[Solver]: total 9.00"],
            "output": "答案: 9.00",
            "passed": True,
            "workflow_stats": {"total_approved": 6},
            "token_stats": {
                "total_prompt_tokens": 100,
                "total_completion_tokens": 20,
            },
        },
        "vib": {
            "prompt": "question",
            "reference_answer": "9",
            "history": ["[Planner]: total 9"],
            "output": "答案: 9",
            "passed": True,
            "workflow_stats": {"total_approved": 2},
            "token_stats": {
                "total_prompt_tokens": 60,
                "total_completion_tokens": 18,
            },
        },
        "error_msg": None,
    }


def test_analyze_collect_trace_examples_reports_coverage():
    module = load_module("analyze_multi_benchmark_test", "experiments/analyze_multi_benchmark.py")

    rows = [
        {
            "task_id": "GSM8K/227",
            "full_passed": True,
            "vib_passed": False,
            "full_messages": 6,
            "vib_messages": 2,
            "full_tokens": 120,
            "vib_tokens": 78,
        }
    ]
    trace_info = module.collect_trace_examples("gsm8k", rows, [sample_trace_case()])

    assert trace_info["trace_coverage"] == 1.0
    assert trace_info["full_only_trace_examples"][0]["task_id"] == "GSM8K/227"


def test_diff_build_case_prefers_same_run_trace():
    module = load_module("diff_benchmark_cases_test", "experiments/diff_benchmark_cases.py")

    case = module.build_case_report_from_trace(sample_trace_case())

    assert case["trace_source"] == "same_run_trace"
    assert case["full"]["output_last_number"] == "9.00"
    assert case["vib"]["messages"] == 2
    assert "VIB 分支消息更少" in case["inferred_root_cause"]


def test_render_markdown_contains_pareto_and_split_sections():
    mod = load_module("analyze_multi_benchmark_2", "experiments/analyze_multi_benchmark.py")

    analyses = [
        {
            "dataset": "MBPP",
            "total_tasks": 257,
            "pass_delta_pp": +1.56,
            "vib_only_count": 11,
            "full_only_count": 7,
            "net_win_count": 4,
            "both_pass_count": 229,
            "both_fail_count": 10,
            "mcnemar_pvalue": 0.21875,
            "same_message_ratio": 0.875,
            "avg_message_delta": 0.01,
            "avg_token_delta": -2.1,
            "full_only_examples": [],
            "vib_only_examples": [],
            "summary": {
                "full_pass_rate": 91.83,
                "vib_pass_rate": 93.39,
                "msg_compression_pct": -0.12,
                "token_saving_pct": -0.05,
                "early_stopped_count": 225,
                "early_stopped_pct": 87.55,
                "early_stopped_full_pass_rate": 93.33,
                "early_stopped_vib_pass_rate": 94.67,
                "early_stopped_pass_delta_pp": +1.34,
                "early_stopped_msg_compression_pct": 0.0,
                "early_stopped_token_saving_pct": 0.0,
                "multi_round_count": 32,
                "multi_round_pct": 12.45,
                "multi_round_full_pass_rate": 81.25,
                "multi_round_vib_pass_rate": 84.38,
                "multi_round_pass_delta_pp": +3.13,
                "multi_round_msg_compression_pct": 22.15,
                "multi_round_token_saving_pct": 5.40,
            },
        }
    ]

    md = mod.render_markdown(Path("results/fake.json"), analyses, {})

    assert "## Pareto 概览" in md
    assert "| Dataset | Total | Full Pass@1" in md
    assert "95% Wilson" in md
    assert "[95% Boot CI]" in md
    assert "## 数据集内部分组拆分" in md
    assert "| MBPP | Early-Stopped |" in md
    assert "| MBPP | Multi-Round |" in md
    assert "225 (87.55%)" in md
    assert "32 (12.45%)" in md


def test_qualitative_classification_buckets():
    mod = load_module("qualitative_case_study", "experiments/generate_qualitative_case_study.py")

    rows = [
        {"task_id": "1", "full_passed": True, "vib_passed": True,
         "full_messages": 6, "vib_messages": 3, "full_tokens": 100, "vib_tokens": 80},
        {"task_id": "2", "full_passed": True, "vib_passed": True,
         "full_messages": 6, "vib_messages": 6, "full_tokens": 100, "vib_tokens": 100},
        {"task_id": "3", "full_passed": False, "vib_passed": True,
         "full_messages": 6, "vib_messages": 4, "full_tokens": 100, "vib_tokens": 85},
        {"task_id": "4", "full_passed": True, "vib_passed": False,
         "full_messages": 6, "vib_messages": 2, "full_tokens": 100, "vib_tokens": 50},
        {"task_id": "5", "full_passed": True, "vib_passed": False,
         "full_messages": 6, "vib_messages": 6, "full_tokens": 100, "vib_tokens": 100},
        {"task_id": "6", "full_passed": True, "vib_passed": False,
         "full_messages": 6, "vib_messages": 4, "full_tokens": 100, "vib_tokens": 80},
    ]

    ds = mod.classify_dataset("toy", rows, [])
    dist = ds["distribution"]

    assert dist["VIB_SUCCESS_SPARSE"]["count"] == 1
    assert dist["BOTH_PASS_TYPICAL"]["count"] == 1
    assert dist["VIB_SUCCESS_NOISE"]["count"] == 1
    assert dist["VIB_FAIL_LOW_MESSAGE"]["count"] == 1
    assert dist["VIB_FAIL_FULL_MSG"]["count"] == 1
    assert dist["VIB_FAIL_MID_MSG"]["count"] == 1

    md = mod.render_markdown(Path("results/fake.json"), [ds])
    assert "# Qualitative Case Study" in md
    assert "VIB_SUCCESS_SPARSE" in md
    assert "VIB_FAIL_LOW_MESSAGE" in md
    assert "跨数据集综合观察" in md
    assert "## 深入讨论：三大基准的结构性损失 vs 随机漂移解耦" in md
    assert "LOW/FULL 比值" in md
    assert "结构性损失判定" in md
    assert "MBPP 指标解耦备注" in md


def test_offline_backfill_split_mode_and_or_produce_correct_counts():
    mod = load_module("offline_backfill", "experiments/offline_backfill_mbpp_split.py")

    summary = {"total_tasks": 5}
    # 构造 5 行 + 5 条 trace：
    # 行 1: f_es=True, v_es=True   → AND=ES, OR=ES
    # 行 2: f_es=True, v_es=False  → AND=MR, OR=ES
    # 行 3: f_es=False, v_es=True  → AND=MR, OR=ES
    # 行 4: f_es=False, v_es=False → AND=MR, OR=MR
    # 行 5: f_es=False, v_es=False → AND=MR, OR=MR
    rows = [
        {"task_id": "1", "full_passed": True, "vib_passed": True, "full_messages": 3, "vib_messages": 3,
         "full_tokens": 300, "vib_tokens": 300, "full_prompt_tokens": 0, "vib_prompt_tokens": 0,
         "full_completion_tokens": 0, "vib_completion_tokens": 0},
        {"task_id": "2", "full_passed": True, "vib_passed": False, "full_messages": 3, "vib_messages": 6,
         "full_tokens": 300, "vib_tokens": 600, "full_prompt_tokens": 0, "vib_prompt_tokens": 0,
         "full_completion_tokens": 0, "vib_completion_tokens": 0},
        {"task_id": "3", "full_passed": False, "vib_passed": True, "full_messages": 6, "vib_messages": 3,
         "full_tokens": 600, "vib_tokens": 300, "full_prompt_tokens": 0, "vib_prompt_tokens": 0,
         "full_completion_tokens": 0, "vib_completion_tokens": 0},
        {"task_id": "4", "full_passed": True, "vib_passed": True, "full_messages": 6, "vib_messages": 5,
         "full_tokens": 600, "vib_tokens": 500, "full_prompt_tokens": 0, "vib_prompt_tokens": 0,
         "full_completion_tokens": 0, "vib_completion_tokens": 0},
        {"task_id": "5", "full_passed": False, "vib_passed": False, "full_messages": 6, "vib_messages": 6,
         "full_tokens": 600, "vib_tokens": 600, "full_prompt_tokens": 0, "vib_prompt_tokens": 0,
         "full_completion_tokens": 0, "vib_completion_tokens": 0},
    ]
    traces = [
        {"task_id": "1", "full": {"early_stopped": True}, "vib": {"early_stopped": True}},
        {"task_id": "2", "full": {"early_stopped": True}, "vib": {"early_stopped": False}},
        {"task_id": "3", "full": {"early_stopped": False}, "vib": {"early_stopped": True}},
        {"task_id": "4", "full": {"early_stopped": False}, "vib": {"early_stopped": False}},
        {"task_id": "5", "full": {"early_stopped": False}, "vib": {"early_stopped": False}},
    ]

    sum_and = {"total_tasks": 5}
    rows_and = [dict(r) for r in rows]
    mod.backfill_dataset_split("MBPP", sum_and, rows_and, traces, split_mode="and")
    assert sum_and["split_mode"] == "and"
    assert sum_and["early_stopped_count"] == 1
    assert sum_and["early_stopped_both_agree_count"] == 1
    assert sum_and["multi_round_count"] == 4
    assert sum_and["early_stopped_pct"] == 20.0
    assert sum_and["multi_round_pct"] == 80.0
    # Early-Stopped 只有行 1 (3/3) → 两侧 100%，压缩 0%
    assert sum_and["early_stopped_full_pass_rate"] == 100.0
    assert sum_and["early_stopped_vib_pass_rate"] == 100.0
    assert sum_and["early_stopped_msg_compression_pct"] == 0.0
    # Multi-Round 4 行 (2-5)：full msgs = 3+6+6+6 = 21；vib msgs = 6+3+5+6 = 20 → (21-20)/21 ≈ 4.76%
    assert sum_and["multi_round_msg_compression_pct"] == round((21 - 20) / 21 * 100, 2)
    # detailed_rows 应回填 early_stopped 字段
    assert rows_and[0]["full_early_stopped"] is True and rows_and[0]["vib_early_stopped"] is True
    assert rows_and[1]["full_early_stopped"] is True and rows_and[1]["vib_early_stopped"] is False
    assert rows_and[3]["full_early_stopped"] is False and rows_and[3]["vib_early_stopped"] is False

    sum_or = {"total_tasks": 5}
    rows_or = [dict(r) for r in rows]
    mod.backfill_dataset_split("MBPP", sum_or, rows_or, traces, split_mode="or")
    assert sum_or["split_mode"] == "or"
    assert sum_or["early_stopped_count"] == 3  # 行 1, 2, 3 至少一侧 early-stop
    assert sum_or["early_stopped_both_agree_count"] == 1
    assert sum_or["multi_round_count"] == 2  # 行 4, 5 两侧都没 early-stop
    # Multi-Round 2 行 (4,5)：full msgs = 12，vib msgs = 11 → (12-11)/12 ≈ 8.33%
    assert sum_or["multi_round_msg_compression_pct"] == round((12 - 11) / 12 * 100, 2)
    # MR: 行 4 双方过，行 5 双方错 → Full 1/2 = 50%，VIB 1/2 = 50%
    assert sum_or["multi_round_full_pass_rate"] == 50.0
    assert sum_or["multi_round_vib_pass_rate"] == 50.0


def test_code_correction_dispatches_humaneval_entry_point(monkeypatch):
    from scripts import recompute_code_eval as correction

    calls = []

    def evaluate(code, tests, entry_point):
        calls.append(entry_point)
        return False

    monkeypatch.setattr(correction.CodeEvaluator, "evaluate", evaluate)
    assert not correction._score("humaneval", {
        "output": "def add(a, b): return a - b",
        "test_cases": ["def check(candidate): assert candidate(1, 2) == 3"],
        "entry_point": "add",
    })
    assert calls == ["add"]


def test_code_correction_dispatches_mbpp_without_entry_point(monkeypatch):
    from scripts import recompute_code_eval as correction

    calls = []

    def evaluate(code, tests, entry_point):
        calls.append(entry_point)
        return True

    monkeypatch.setattr(correction.CodeEvaluator, "evaluate", evaluate)
    assert correction._score("mbpp", {
        "output": "def add(a, b): return a + b",
        "test_cases": ["assert add(1, 2) == 3"],
        "entry_point": "add",
    })
    assert calls == [None]


def test_code_correction_refreshes_mbpp_split_without_changing_outputs(monkeypatch):
    from scripts import recompute_code_eval as correction
    import copy

    traces = [{
        "task_id": "MBPP/1",
        "full": {"passed": False, "early_stopped": False, "output": "full"},
        "vib": {"passed": True, "early_stopped": True, "output": "vib"},
    }]
    rows = [{
        "task_id": "MBPP/1", "full_passed": False, "vib_passed": True,
        "full_messages": 6, "vib_messages": 3, "full_tokens": 100, "vib_tokens": 80,
    }]
    data = {
        "trace_results": {"mbpp": traces},
        "detailed_results": {"mbpp": rows},
        "summary": {"mbpp": {
            "total_tasks": 1, "same_message_ratio": 0.0, "avg_message_delta": 3.0,
            "full_tokens": 100, "vib_tokens": 80,
        }},
    }
    before = copy.deepcopy(traces)
    monkeypatch.setattr(correction, "_score", lambda dataset, trace: True)
    report = correction._recompute_dataset("mbpp", data)
    assert report["flip_count"] == 1
    assert data["summary"]["mbpp"]["full_pass_rate"] == 100.0
    assert data["summary"]["mbpp"]["early_stopped_full_pass_rate"] == 100.0
    assert data["summary"]["mbpp"]["early_stopped_count"] == 1
    assert data["summary"]["mbpp"]["multi_round_count"] == 0
    assert data["summary"]["mbpp"]["full_tokens"] == 100
    assert rows[0]["full_passed"] is True
    for arm in ("full", "vib"):
        assert traces[0][arm]["output"] == before[0][arm]["output"]
        assert traces[0][arm]["early_stopped"] == before[0][arm]["early_stopped"]

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

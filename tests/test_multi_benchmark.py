import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "experiments" / "run_multi_benchmark.py"


def load_run_multi_benchmark_module():
    spec = importlib.util.spec_from_file_location("run_multi_benchmark_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gsm8k_math_evaluator_accepts_numeric_equivalence():
    module = load_run_multi_benchmark_module()

    assert module.MathEvaluator.evaluate("答案: 9.00", "9")
    assert module.MathEvaluator.evaluate("The answer is 1,234.0", "1234")
    assert module.MathEvaluator.evaluate("答案: -2.50", "-2.5")
    assert not module.MathEvaluator.evaluate("答案: 8", "9")


def test_benchmark_report_includes_same_run_trace_results(monkeypatch):
    module = load_run_multi_benchmark_module()

    sample = module.TaskSample(
        task_id="GSM8K/test",
        dataset_name="GSM8K",
        prompt="What is 4 + 5?",
        reference_answer="9"
    )

    class FakeAdapter:
        def load(self, limit=-1):
            return [sample]

    monkeypatch.setattr(module.DatasetFactory, "get_adapter", lambda name: FakeAdapter())

    harness = module.MultiDatasetHarness(api_key="dummy-key", workers=1)

    fake_result = module.TaskResult(
        task_id=sample.task_id,
        dataset_name=sample.dataset_name,
        full_passed=True,
        vib_passed=True,
        full_messages=2,
        vib_messages=1,
        full_tokens=20,
        vib_tokens=10,
        execution_time=0.5,
        full_prompt_tokens=12,
        full_completion_tokens=8,
        vib_prompt_tokens=6,
        vib_completion_tokens=4
    )

    monkeypatch.setattr(
        harness,
        "run_single_task_with_trace",
        lambda s: {
            "task_result": fake_result,
            "full": {
                "history": ["[Planner]: 4 + 5 = 9"],
                "workflow_stats": {"total_approved": 2},
                "output": "答案: 9",
                "passed": True,
                "token_stats": {
                    "total_calls": 1,
                    "total_prompt_tokens": 12,
                    "total_completion_tokens": 8,
                    "total_tokens": 20
                }
            },
            "vib": {
                "history": ["[Planner]: 9"],
                "workflow_stats": {"total_approved": 1},
                "output": "答案: 9.00",
                "passed": True,
                "token_stats": {
                    "total_calls": 1,
                    "total_prompt_tokens": 6,
                    "total_completion_tokens": 4,
                    "total_tokens": 10
                }
            }
        }
    )

    report = harness.run_benchmark(["gsm8k"], limit=1)

    assert "trace_results" in report
    assert "gsm8k" in report["trace_results"]
    trace = report["trace_results"]["gsm8k"][0]
    assert trace["task_id"] == sample.task_id
    assert trace["full"]["history"] == ["[Planner]: 4 + 5 = 9"]
    assert trace["vib"]["output"] == "答案: 9.00"

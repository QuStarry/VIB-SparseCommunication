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


def test_build_final_code_prompt_prioritizes_problem_examples():
    module = load_run_multi_benchmark_module()

    prompt = (
        "def foo(x):\n"
        "    \"\"\"Add one.\n"
        "    >>> foo(1)\n"
        "    2\n"
        "    >>> foo(5)\n"
        "    6\n"
        "    \"\"\""
    )
    history = [
        "[Architect]: 先考虑边界。",
        "[Reviewer]: 如果讨论冲突，应以题面和示例为准。"
    ]
    final_prompt = module.build_final_code_prompt(prompt, history)

    assert "docstring 示例和隐含测试语义为最高优先级" in final_prompt
    assert "优先选择最直接、最贴合示例的正确实现" in final_prompt
    assert "忽略讨论中的相反结论" in final_prompt
    assert "以题面为最终裁决" in final_prompt
    assert "【题面可见示例】" in final_prompt
    assert ">>> foo(1)\n2" in final_prompt
    assert prompt in final_prompt


def test_extract_visible_examples_collects_doctest_blocks():
    module = load_run_multi_benchmark_module()

    prompt = (
        "def foo(x):\n"
        "    \"\"\"Add one.\n"
        "    >>> foo(1)\n"
        "    2\n"
        "\n"
        "    >>> foo(5)\n"
        "    6\n"
        "    \"\"\""
    )

    examples = module.extract_visible_examples(prompt)
    assert examples == [">>> foo(1)\n2", ">>> foo(5)\n6"]


def test_extract_visible_examples_collects_input_output_and_arrow_examples():
    module = load_run_multi_benchmark_module()

    prompt = (
        "def maximum(arr, k):\n"
        "    \"\"\"\n"
        "    Example 1:\n"
        "        Input: arr = [-3, -4, 5], k = 3\n"
        "        Output: [-4, -3, 5]\n"
        "    Example 2:\n"
        "        Input: arr = [4, -4, 4], k = 2\n"
        "        Output: [4, 4]\n"
        "    specialFilter([15, -73, 14, -15]) => 1\n"
        "    split_words(\"abcdef\") == 3\n"
        "    check_if_last_char_is_a_letter(\"apple pi e\") ➞ True\n"
        "    \"\"\"\n"
    )

    examples = module.extract_visible_examples(prompt)
    assert "Input: arr = [-3, -4, 5], k = 3\nOutput: [-4, -3, 5]" in examples
    assert "Input: arr = [4, -4, 4], k = 2\nOutput: [4, 4]" in examples
    assert "specialFilter([15, -73, 14, -15]) => 1" in examples
    assert "split_words(\"abcdef\") == 3" in examples
    assert "check_if_last_char_is_a_letter(\"apple pi e\") ➞ True" in examples


def test_generate_final_code_retries_on_api_error():
    module = load_run_multi_benchmark_module()

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            if self.calls < 3:
                return {"content": "[API Error: Request timed out.]"}
            return {"content": "```python\ndef foo():\n    return 1\n```"}

    client = FakeClient()
    code = module.generate_final_code(client, "def foo(): pass", ["[Coder]: return 1"])

    assert client.calls == 3
    assert code == "def foo():\n    return 1"

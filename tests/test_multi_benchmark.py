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
    assert "不要添加题面和示例均未明确要求的防御性分支或额外边界校验" in final_prompt
    assert "题面没有写应该抛异常时永远不要写 raise 分支" in final_prompt
    assert "永远不要因为“我认为示例有误”" in final_prompt
    assert "若 Reviewer 明确给出“形如 输入X → 输出Y”的单点断言" in final_prompt
    assert "有明确多数派（出现次数 ≥ 其他方案之和）时选多数派" in final_prompt
    assert "任何被 Reviewer 用具体反例明确否决的方案" in final_prompt
    assert "Reviewer 给出的具体反例（如“shift = 位数时应返回原字符串" in final_prompt
    assert "严格照抄题面文本里的运算符用词" in final_prompt
    assert "题面没提到应 raise ValueError 时，永远不要写 raise 分支" in final_prompt


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


def test_benchmark_summary_splits_early_stopped_and_multi_round(monkeypatch):
    module = load_run_multi_benchmark_module()

    samples = [
        module.TaskSample(task_id="MBPP/es1", dataset_name="MBPP", prompt="p1", reference_answer="a1"),
        module.TaskSample(task_id="MBPP/es2", dataset_name="MBPP", prompt="p2", reference_answer="a2"),
        module.TaskSample(task_id="MBPP/mr1", dataset_name="MBPP", prompt="p3", reference_answer="a3"),
    ]

    class FakeAdapter:
        def load(self, limit=-1):
            return samples

    monkeypatch.setattr(module.DatasetFactory, "get_adapter", lambda name: FakeAdapter())

    harness = module.MultiDatasetHarness(api_key="dummy", workers=1)

    fake_results = [
        module.TaskResult(
            task_id="MBPP/es1", dataset_name="MBPP",
            full_passed=True, vib_passed=True,
            full_messages=3, vib_messages=3,
            full_tokens=300, vib_tokens=300,
            execution_time=1.0,
            full_prompt_tokens=200, full_completion_tokens=100,
            vib_prompt_tokens=200, vib_completion_tokens=100,
            full_early_stopped=True, vib_early_stopped=True,
        ),
        module.TaskResult(
            task_id="MBPP/es2", dataset_name="MBPP",
            full_passed=False, vib_passed=True,
            full_messages=4, vib_messages=3,
            full_tokens=400, vib_tokens=350,
            execution_time=1.0,
            full_prompt_tokens=250, full_completion_tokens=150,
            vib_prompt_tokens=220, vib_completion_tokens=130,
            full_early_stopped=False, vib_early_stopped=True,
        ),
        module.TaskResult(
            task_id="MBPP/mr1", dataset_name="MBPP",
            full_passed=True, vib_passed=True,
            full_messages=6, vib_messages=4,
            full_tokens=800, vib_tokens=700,
            execution_time=1.0,
            full_prompt_tokens=500, full_completion_tokens=300,
            vib_prompt_tokens=430, vib_completion_tokens=270,
            full_early_stopped=False, vib_early_stopped=False,
        ),
    ]

    def _fake_run(sample):
        idx = next(i for i, s in enumerate(samples) if s.task_id == sample.task_id)
        res = fake_results[idx]
        empty_wf = {
            "history": [],
            "workflow_stats": {"total_approved": res.full_messages},
            "output": "ok",
            "passed": res.full_passed,
            "token_stats": {"total_prompt_tokens": res.full_prompt_tokens,
                            "total_completion_tokens": res.full_completion_tokens,
                            "total_tokens": res.full_tokens},
        }
        empty_wv = {
            "history": [],
            "workflow_stats": {"total_approved": res.vib_messages},
            "output": "ok",
            "passed": res.vib_passed,
            "token_stats": {"total_prompt_tokens": res.vib_prompt_tokens,
                            "total_completion_tokens": res.vib_completion_tokens,
                            "total_tokens": res.vib_tokens},
        }
        return {"task_result": res, "full": empty_wf, "vib": empty_wv}

    monkeypatch.setattr(harness, "run_single_task_with_trace", _fake_run)
    report = harness.run_benchmark(["mbpp"], limit=3)
    summary = report["summary"]["mbpp"]

    assert summary["early_stopped_count"] == 2
    assert summary["early_stopped_pct"] == round(2 / 3 * 100, 2)
    assert summary["early_stopped_both_agree_count"] == 1
    assert summary["multi_round_count"] == 1
    assert summary["multi_round_pct"] == round(1 / 3 * 100, 2)
    assert summary["early_stopped_full_pass_rate"] == 50.0
    assert summary["early_stopped_vib_pass_rate"] == 100.0
    assert summary["early_stopped_pass_delta_pp"] == +50.0
    assert summary["multi_round_msg_compression_pct"] == round((6 - 4) / 6 * 100, 2)
    assert summary["multi_round_token_saving_pct"] == round((800 - 700) / 800 * 100, 2)
    assert summary["early_stopped_same_message_ratio"] == round(1 / 2, 4)
    assert summary["multi_round_same_message_ratio"] == 0.0


SAME_CHARS_54_PROMPT = '''
def same_chars(s0: str, s1: str):
    """
    Check if two words have the same characters.
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
    True
    >>> same_chars('abcd', 'dddddddabc')
    True
    >>> same_chars('dddddddabc', 'abcd')
    True
    >>> same_chars('eabcd', 'dddddddabc')
    False
    >>> same_chars('abcd', 'dddddddabce')
    False
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddddabc')
    False
    """
'''


def test_run_visible_doctest_detects_counter_vs_set_mismatch_on_humaneval_54():
    module = load_run_multi_benchmark_module()
    counter_code = '''def same_chars(s0: str, s1: str):
    from collections import Counter
    return Counter(s0) == Counter(s1)'''
    set_code = '''def same_chars(s0: str, s1: str):
    return set(s0) == set(s1)'''
    counter_failures = module._run_visible_doctest_against_code(counter_code, SAME_CHARS_54_PROMPT)
    set_failures = module._run_visible_doctest_against_code(set_code, SAME_CHARS_54_PROMPT)
    assert len(set_failures) == 0
    assert len(counter_failures) == 3
    assert any("eabcdzzzz" in f["args_repr"] and f["want_repr"] == "True" for f in counter_failures)
    assert any("'abcd', 'dddddddabc'" in f["args_repr"] or "('abcd', 'dddddddabc')" in f["args_repr"] for f in counter_failures)


def test_build_final_code_prompt_includes_step0_doctest_filter_and_permanent_elimination():
    module = load_run_multi_benchmark_module()
    p = module.build_final_code_prompt(SAME_CHARS_54_PROMPT, [
        "[Reviewer]: 该实现需确保字符多重集合相等，而非仅字符种类相同。建议使用 Counter 比较。",
    ])
    assert "第 0 步 强制算法过滤器" in p
    assert "永久淘汰" in p
    assert "全通候选" in p
    assert "具体数值/语义代入计算" in p


def test_generate_final_code_retries_when_visible_doctest_fails(monkeypatch):
    module = load_run_multi_benchmark_module()
    monkeypatch.setattr(module, "_run_visible_doctest_against_code", lambda code, prompt: [
        {"args_repr": "same_chars('a','a')", "want_repr": "True", "got_repr": "False",
         "doctest_line_repr": "same_chars('a','a') -> True"},
    ] if "Counter" in code else [])

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"content": "```python\ndef same_chars(s0, s1):\n  from collections import Counter\n  return Counter(s0) == Counter(s1)\n```"}
            return {"content": "```python\ndef same_chars(s0, s1):\n  return set(s0) == set(s1)\n```"}

    client = FakeClient()
    code = module.generate_final_code(client, SAME_CHARS_54_PROMPT, [])
    assert client.calls == 2
    assert "set(s0) == set(s1)" in code


def test_generate_final_code_injects_doctest_failures_into_retry_prompt(monkeypatch):
    module = load_run_multi_benchmark_module()

    def _fake_doctest(code, prompt):
        if "BadCounter" in code and "AlsoBad" not in code:
            return [{"args_repr": "same_chars('a','b')", "want_repr": "False", "got_repr": "True",
                     "doctest_line_repr": "same_chars('a','b') -> False"}]
        if "AlsoBad" in code:
            return [{"args_repr": "same_chars('a','a')", "want_repr": "True", "got_repr": "False",
                     "doctest_line_repr": "same_chars('a','a') -> True"}]
        return []

    monkeypatch.setattr(module, "_run_visible_doctest_against_code", _fake_doctest)

    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.message_contents = []

        def chat(self, **kwargs):
            self.calls += 1
            msgs = kwargs.get("messages") or []
            content = msgs[-1].get("content", "") if msgs else ""
            self.message_contents.append(content)
            if self.calls == 1:
                return {"content": "```python\ndef same_chars(s0, s1):\n  return BadCounter(s0)==BadCounter(s1)\n```"}
            if self.calls == 2:
                return {"content": "```python\ndef same_chars(s0, s1):\n  return AlsoBad(s0)==AlsoBad(s1)\n```"}
            return {"content": "```python\ndef same_chars(s0, s1):\n  return set(s0)==set(s1)\n```"}

    client = FakeClient()
    code = module.generate_final_code(client, SAME_CHARS_54_PROMPT, [])
    assert client.calls >= 2
    # 第 2 次 prompt 必须包含第 1 次 doctest 的真实失败明细（BadCounter→same_chars('a','b') 那一条）
    second_prompt = client.message_contents[1]
    assert "上一版代码在题面 doctest 上的真实执行失败" in second_prompt
    assert "same_chars('a','b')" in second_prompt
    assert "你的代码输出 True" in second_prompt
    assert "题面期望 False" in second_prompt
    # 第 3 次 prompt 必须包含 AlsoBad 的失败明细（same_chars('a','a') 那条）
    if len(client.message_contents) >= 3:
        third_prompt = client.message_contents[2]
        assert "same_chars('a','a')" in third_prompt
    assert "set(s0)==set(s1)" in code


def test_generate_final_code_runs_doctest_on_all_3_attempts(monkeypatch):
    """HIGH-5 修复后：每次 attempt（包括第 3 次 = attempt=2）都必须调用 doctest 验证。"""
    module = load_run_multi_benchmark_module()
    doctest_call_count = {"n": 0}
    always_fail = [{"args_repr": "f()", "want_repr": "1", "got_repr": "2", "doctest_line_repr": "f()->1"}]

    def _spying_fail_code(code, prompt):
        doctest_call_count["n"] += 1
        return list(always_fail)

    monkeypatch.setattr(module, "_run_visible_doctest_against_code", _spying_fail_code)

    class FakeClient:
        def __init__(self):
            self.calls = 0
        def chat(self, **kwargs):
            self.calls += 1
            return {"content": f"```python\ndef same_chars(s0,s1):\n  return #{self.calls}-bad\n```"}

    client = FakeClient()
    module.generate_final_code(client, SAME_CHARS_54_PROMPT, [])
    assert client.calls == 3
    # 3 次 attempt 每次都应触发 doctest 检查
    assert doctest_call_count["n"] == 3


def test_generate_final_code_picks_best_candidate_when_all_3_fail(monkeypatch):
    """HIGH-5：当 3 次 attempt 全部 doctest 失败时，返回失败次数最少（最优）的那次代码，而非盲信最后一次。"""
    module = load_run_multi_benchmark_module()
    fail_counts_per_attempt = [3, 1, 4]
    generated_codes = []

    def _varied_failure(code, prompt):
        attempt_idx = len(generated_codes) - 1
        fails = fail_counts_per_attempt[attempt_idx]
        return [{"args_repr": f"f{i}()", "want_repr": "1", "got_repr": "0",
                 "doctest_line_repr": f"f{i}()->1"} for i in range(fails)]

    monkeypatch.setattr(module, "_run_visible_doctest_against_code", _varied_failure)

    class FakeClient:
        def __init__(self):
            self.calls = 0
        def chat(self, **kwargs):
            self.calls += 1
            code = f"def same_chars(s0,s1):\n    return ATTEMPT_{self.calls}_CODE\n"
            generated_codes.append(code)
            return {"content": f"```python\n{code}\n```"}

    client = FakeClient()
    result = module.generate_final_code(client, SAME_CHARS_54_PROMPT, [])
    assert client.calls == 3
    # 3 次 attempt 的 doctest 失败数：[3, 1, 4] → 最少是第 2 次（attempt=1）
    assert "ATTEMPT_2_CODE" in result
    assert "ATTEMPT_3_CODE" not in result

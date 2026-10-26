import json
from types import SimpleNamespace

import pytest

from experiments import run_frozen_replay as replay


@pytest.mark.parametrize("output,reference,expected", [
    ("答案: 9.00", "9", True),
    ("答案: -2.5", "-2.50", True),
    ("\\boxed{9}", "9", True),
    ("答案: 19", "9", False),
    ("答案: 1/9", "9", False),
    ("答案: 12,10,6", "6", False),
    ("推导得到 9，但答案未知", "9", False),
    ("答案: x^9", "9", False),
])
def test_strict_scalar_equal(output, reference, expected):
    assert replay.strict_scalar_equal(output, reference) is expected


def trace(task_id):
    return {
        "task_id": task_id,
        "full": {
            "reference_answer": "9", "prompt": "What is 4 + 5?",
            "history": [
                f"[{role}]: 4 + 5 = 9"
                for _ in range(2) for role in ("Planner", "Solver", "Verifier")
            ],
        },
    }


def code_trace(task_id):
    return {
        "task_id": task_id,
        "full": {
            "prompt": "def add(a, b):\n    '''Return a + b.'''\n",
            "entry_point": "add",
            "test_cases": ["def check(candidate):\n    assert candidate(1, 2) == 3"],
            "history": [
                f"[{role}]: 直接返回 a + b"
                for _ in range(2) for role in ("Architect", "Coder", "Reviewer")
            ],
        },
    }


def test_selection_is_independent_of_historical_outcomes():
    rows = [trace(f"MATH/{index}") for index in range(10)]
    selected = replay.select_tasks(rows, 4, 1, "math")
    for row in rows:
        row["full"]["passed"] = False
    assert [row["task_id"] for row in selected] == [
        row["task_id"] for row in replay.select_tasks(list(reversed(rows)), 4, 1, "math")
    ]


def test_selection_rejects_duplicate_tasks():
    with pytest.raises(ValueError, match="task_id"):
        replay.select_tasks([trace("MATH/1"), trace("MATH/1")], 1, 1, "math")


def test_selection_is_dataset_specific():
    mixed = [trace("MATH/1"), code_trace("HumanEval/1")]
    assert [row["task_id"] for row in replay.select_tasks(mixed, 1, 1, "humaneval")] == [
        "HumanEval/1"
    ]
    # 代码题没有整数参考答案，不能混进数学子集。
    assert [row["task_id"] for row in replay.select_tasks(mixed, 1, 1, "math")] == ["MATH/1"]
    with pytest.raises(ValueError, match="数据集"):
        replay.select_tasks(mixed, 1, 1, "mbpp")


def test_code_scoring_runs_the_task_tests():
    task = replay.build_task(code_trace("HumanEval/1"), "humaneval")
    assert replay.score_output("humaneval", task, "def add(a, b):\n    return a + b\n")
    assert not replay.score_output("humaneval", task, "def add(a, b):\n    return a - b\n")


def test_code_prompt_does_not_contain_tests():
    task = replay.build_task(code_trace("HumanEval/1"), "humaneval")
    prompt = replay.build_prompt("humaneval", task, task["candidates"])
    assert "def check" not in prompt
    assert "assert" not in prompt


def test_gate_uses_frozen_history_and_full_message(monkeypatch):
    lengths = []

    class Router:
        derive_task_seed = staticmethod(lambda seed, salt: seed)

        def __init__(self, **kwargs):
            assert kwargs["beta_0"] == 3.2
            assert kwargs["temp_tau"] == 0.5
            assert kwargs["gamma"] == -0.4
            assert kwargs["adaptive"] is True
            assert kwargs["enable_math_critical_guard"] is True

        def evaluate_detailed(self, candidate_msg, history, **kwargs):
            lengths.append(len(history))
            from src.vib_router import VIBEvaluationResult
            return VIBEvaluationResult(1, 1, 0, 1, True, 0)

    monkeypatch.setattr(replay, "VIBSparseCommRouter", Router)
    messages = trace("MATH/1")["full"]["history"]
    admitted, decisions = replay.admitted_history(messages, "goal", "vib", "id", 1)
    assert admitted == messages
    assert lengths == [0, 0, 0, 3, 3, 3]
    assert [row["candidate_index"] for row in decisions] == list(range(6))


def test_prompt_does_not_contain_reference():
    prompt = replay.synthesis_prompt("What is 4 + 5?", [])
    assert "reference" not in prompt
    assert "答案: 9" not in prompt


def arguments(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"trace_results": {"math": [trace("MATH/1")]}}))
    return SimpleNamespace(
        source=source, dataset="math", tasks=1, repeats=1, seed=1, model="fake",
        provider="fake", input_rate=0.3, output_rate=1.2, max_usd=3,
        output=tmp_path / "run", dry_run=False,
    )


def fake_client(monkeypatch, fail=False):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if fail:
            raise TimeoutError()
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=10,
                model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 10},
            ),
            model="fake-response", id="response-id",
            choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="答案: 9")
            )],
        )

    endpoint = SimpleNamespace(
        with_options=lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
    )
    monkeypatch.setattr(replay, "build_client", lambda *args: SimpleNamespace(client=endpoint, model="fake"))
    monkeypatch.setattr(replay, "admitted_history", lambda messages, *args: (messages, []))
    return calls


def test_completed_requests_are_not_repeated(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    calls = fake_client(monkeypatch)
    replay.run(args)
    replay.run(args)
    assert len(calls) == 6
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["stop_reason"] == "complete"
    assert summary["proxy_usd"] == pytest.approx(6 * 0.000042)


def test_failure_preserves_reservation_and_blocks_resume(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    calls = fake_client(monkeypatch, fail=True)
    with pytest.raises(RuntimeError):
        replay.run(args)
    with pytest.raises(RuntimeError):
        replay.run(args)
    assert len(calls) == 1
    record = json.loads(next(args.output.glob("request_*.json")).read_text())
    assert record["status"] == "error"
    assert record["charged_proxy_usd"] > 0


def test_budget_stops_before_network_request(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.max_usd = 0.000001
    calls = fake_client(monkeypatch)
    replay.run(args)
    assert not calls
    assert json.loads((args.output / "summary.json").read_text())["stop_reason"] == "budget_limit"


def test_changed_protocol_cannot_resume(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    fake_client(monkeypatch)
    replay.run(args)
    args.seed += 1
    with pytest.raises(ValueError, match="协议"):
        replay.run(args)


def test_analysis_rejects_incomplete_pairing(tmp_path, monkeypatch):
    from experiments.analyze_frozen_replay import analyze
    args = arguments(tmp_path)
    fake_client(monkeypatch)
    replay.run(args)
    next(args.output.glob("request_*.json")).unlink()
    with pytest.raises(ValueError, match="配对"):
        analyze(args.output)


def test_analysis_aggregates_repeats_by_task(tmp_path, monkeypatch):
    from experiments.analyze_frozen_replay import analyze
    args = arguments(tmp_path)
    args.repeats = 2
    fake_client(monkeypatch)
    replay.run(args)
    report = analyze(args.output)
    assert report["valid_requests"] == 12
    assert report["independent_tasks"] == 1
    assert report["vib_first_round_identical_prompts"] == 2
    assert report["paired_comparisons"]["vib_minus_full"]["delta_pp"] == 0
    json.dumps(report)


def test_analysis_rejects_default_router_protocol(tmp_path, monkeypatch):
    from experiments.analyze_frozen_replay import analyze
    args = arguments(tmp_path)
    fake_client(monkeypatch)
    replay.run(args)
    path = args.output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["version"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="论文配置"):
        analyze(args.output)


def test_analysis_rejects_changed_request_identity(tmp_path, monkeypatch):
    from experiments.analyze_frozen_replay import analyze
    args = arguments(tmp_path)
    fake_client(monkeypatch)
    replay.run(args)
    path = next(args.output.glob("request_*.json"))
    record = json.loads(path.read_text())
    record["request_id"] = "wrong-contract"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="契约"):
        analyze(args.output)


def test_boundary_features_ignore_reference_material():
    from experiments.analyze_frozen_replay import history_features
    task = replay.build_task(code_trace("HumanEval/1"), "humaneval")
    before = history_features(task["candidates"])
    task["entry_point"] = "hidden"
    task["test_cases"] = ["raise RuntimeError('must not execute')"]
    assert before == history_features(task["candidates"])
    assert before["candidate_messages"] == 6


def test_boundary_features_normalize_code_without_execution():
    from experiments.analyze_frozen_replay import history_features
    messages = [
        "[Coder]: ```python\nraise RuntimeError('must not execute')\n```",
        "[Reviewer]: ```python\n# same AST\nraise RuntimeError('must not execute')\n```",
        "[Architect]: ```python\nthis is not valid python!\n```",
        "[Coder]: ```python\nvalue = 2\n```",
        "[Reviewer]: additional evidence",
        "[Architect]: additional evidence",
    ]
    features = history_features(messages)
    assert features["python_fence_blocks"] == 4
    assert features["parsable_python_blocks"] == 3
    assert features["distinct_python_ast_count"] == 2
    assert 0 <= features["second_round_new_word_fraction"] <= 1


def test_boundary_features_reject_incomplete_rounds():
    from experiments.analyze_frozen_replay import history_features
    with pytest.raises(ValueError, match="六候选"):
        history_features(["partial"])


def test_boundary_preserves_opposite_repeat_outcomes():
    from experiments.analyze_frozen_replay import compare_policies
    lookup = {}
    for repeat, full_passed in enumerate((True, False)):
        lookup["task", repeat, "full"] = {
            "passed": full_passed, "prompt_sha256": "same",
        }
        lookup["task", repeat, "first_round"] = {
            "passed": not full_passed, "prompt_sha256": "same",
        }
    report = compare_policies(["task"], 2, lookup, "full", "first_round")
    assert report["delta_pp"] == 0
    assert report["independent_tasks"] == 1
    assert report["tasks_tied"] == 1
    assert report["tasks_with_opposite_repeat_differences"] == 1
    assert report["identical_prompt_score_disagreements"] == 2
    assert report["tasks"][0]["repeat_outcomes"]["policy_only_pass"] == 1
    assert report["tasks"][0]["repeat_outcomes"]["baseline_only_pass"] == 1


def test_boundary_reports_observed_harm_with_distinct_prompts():
    from experiments.analyze_frozen_replay import compare_policies
    lookup = {
        ("task", 0, "full"): {"passed": False, "prompt_sha256": "full"},
        ("task", 0, "first_round"): {"passed": True, "prompt_sha256": "first"},
    }
    report = compare_policies(["task"], 1, lookup, "full", "first_round")
    assert report["delta_pp"] == -100
    assert report["tasks_worse"] == 1
    assert report["identical_prompt_pairs"] == 0
    assert report["tasks"][0]["observed_direction"] == "worse"


def boundary_arguments(tmp_path):
    args = arguments(tmp_path)
    prior = tmp_path / "prior_manifest.json"
    prior.write_text(json.dumps({"dataset": "math", "tasks": [{"task_id": "MATH/old"}]}))
    args.protocol = "boundary"
    args.model = "fake-response"
    args.exclude_manifest = [prior]
    return args


def test_boundary_selection_excludes_old_pilot_tasks():
    rows = [trace("MATH/old"), trace("MATH/new")]
    selected = replay.select_tasks(rows, 1, 1, "math", {"MATH/old"})
    assert [row["task_id"] for row in selected] == ["MATH/new"]


def test_boundary_control_uses_identical_first_round_history():
    messages = trace("MATH/1")["full"]["history"]
    assert replay.admitted_history(messages, "goal", "first_round_control", "id", 1) == (
        replay.admitted_history(messages, "goal", "first_round", "id", 1)
    )


def test_boundary_runs_four_policies_and_resumes_without_calls(tmp_path, monkeypatch):
    from experiments.analyze_frozen_replay import analyze
    args = boundary_arguments(tmp_path)
    args.repeats = 2
    calls = fake_client(monkeypatch)
    replay.run(args)
    replay.run(args)
    assert len(calls) == 8
    report = analyze(args.output)
    assert set(report["policies"]) == set(replay.BOUNDARY_POLICIES)
    control = report["collaboration_boundary"]["contrasts"][
        "first_round_control_minus_first_round"
    ]
    assert control["identical_prompt_pairs"] == 2
    assert control["independent_tasks"] == 1
    assert control["identical_prompt_score_disagreements"] == 0


def test_boundary_requires_explicit_exclusion_manifest(tmp_path, monkeypatch):
    args = boundary_arguments(tmp_path)
    args.exclude_manifest = []
    calls = fake_client(monkeypatch)
    with pytest.raises(ValueError, match="排除"):
        replay.run(args)
    assert not calls


def test_boundary_model_drift_stops_and_blocks_paid_resume(tmp_path, monkeypatch):
    args = boundary_arguments(tmp_path)
    args.model = "different-pinned-model"
    calls = fake_client(monkeypatch)
    with pytest.raises(RuntimeError, match="请求失败"):
        replay.run(args)
    with pytest.raises(RuntimeError, match="人工审计"):
        replay.run(args)
    assert len(calls) == 1


def test_boundary_analysis_rejects_nonidentical_noise_control(tmp_path, monkeypatch):
    from experiments.analyze_frozen_replay import analyze
    args = boundary_arguments(tmp_path)
    fake_client(monkeypatch)
    replay.run(args)
    for path in args.output.glob("request_*.json"):
        record = json.loads(path.read_text())
        if record["policy"] == "first_round_control":
            record["prompt_sha256"] = "different-prompt"
            path.write_text(json.dumps(record))
            break
    with pytest.raises(ValueError, match="提示词不相同"):
        analyze(args.output)


def test_seeded_code_scoring_preserves_parent_random_state():
    import random
    task = replay.build_task(code_trace("HumanEval/1"), "humaneval")
    task["test_cases"] = [
        "def check(candidate):"
        "\n    import random"
        "\n    assert candidate(1, 2) == 3"
        "\n    assert random.random() == 0.13436424411240122"
    ]
    state = random.getstate()
    try:
        assert replay.score_output("humaneval", task, "def add(a, b): return a + b", 1)
        assert random.getstate() == state
    finally:
        random.setstate(state)


def test_seeded_code_scoring_rejects_inconsistent_independent_checks(monkeypatch):
    from experiments.run_multi_benchmark import CodeEvaluator
    results = iter((True, False))
    monkeypatch.setattr(CodeEvaluator, "evaluate", lambda *args, **kwargs: next(results))
    task = replay.build_task(code_trace("HumanEval/1"), "humaneval")
    with pytest.raises(ValueError, match="不一致"):
        replay.score_output("humaneval", task, "def add(a, b): return a + b", 1)

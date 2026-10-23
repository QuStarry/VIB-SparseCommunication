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


def test_selection_is_independent_of_historical_outcomes():
    rows = [trace(f"MATH/{index}") for index in range(10)]
    selected = replay.select_tasks(rows, 4, 1)
    for row in rows:
        row["full"]["passed"] = False
    assert [row["task_id"] for row in selected] == [
        row["task_id"] for row in replay.select_tasks(list(reversed(rows)), 4, 1)
    ]


def test_selection_rejects_duplicate_tasks():
    with pytest.raises(ValueError, match="task_id"):
        replay.select_tasks([trace("MATH/1"), trace("MATH/1")], 1, 1)


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
        source=source, tasks=1, repeats=1, seed=1, model="fake", provider="fake",
        input_rate=0.3, output_rate=1.2, max_usd=3, output=tmp_path / "run",
        dry_run=False,
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

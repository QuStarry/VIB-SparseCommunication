from types import SimpleNamespace
import json
import sys

import pytest

from src.llm_client import LLMClient, build_client
from src.llm_agents import LLMAgent
from experiments import run_multi_benchmark as benchmark


def recording_client(**kwargs):
    client = LLMClient(api_key="test-key", base_url="https://example.invalid/v1", **kwargs)
    calls = []

    def create(**request):
        calls.append(request)
        return SimpleNamespace(
            id="test-response",
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            choices=[SimpleNamespace(message=SimpleNamespace(content="def f(): return 1"))],
        )

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


@pytest.mark.parametrize("temperature", [0, 0.43])
def test_explicit_global_parameters_reach_all_stages(temperature):
    client, calls = recording_client(temperature=temperature, max_tokens=77)
    LLMAgent("Coder", "Write code.", client).generate("goal", [], 1)
    benchmark.generate_final_code(client, "def f(): pass", [])
    benchmark.generate_final_answer(client, "1+1", [])
    assert [(call["temperature"], call["max_tokens"]) for call in calls] == [
        (temperature, 77), (temperature, 77), (temperature, 77),
    ]


def test_omitted_global_parameters_preserve_historical_stage_defaults():
    client, calls = recording_client()
    LLMAgent("Coder", "Write code.", client).generate("goal", [], 1)
    benchmark.generate_final_code(client, "def f(): pass", [])
    benchmark.generate_final_answer(client, "1+1", [])
    assert [(call["temperature"], call["max_tokens"]) for call in calls] == [
        (0.2, 256), (0.1, 1024), (0.1, 256),
    ]


def test_stage_override_takes_precedence_over_global_configuration():
    from src.inference_protocol import InferenceProtocol

    client, calls = recording_client(temperature=0, max_tokens=77)
    client.inference_protocol = InferenceProtocol(
        temperature=0, max_tokens=77, agent_temperature=0.3, math_max_tokens=99,
    )
    LLMAgent("Coder", "Write code.", client).generate("goal", [], 1)
    benchmark.generate_final_answer(client, "1+1", [])
    assert [(call["temperature"], call["max_tokens"]) for call in calls] == [(0.3, 77), (0, 99)]


def test_explicit_agent_call_override_remains_supported():
    client, calls = recording_client(temperature=0, max_tokens=77)
    LLMAgent("Coder", "Write code.", client, max_tokens=45, temperature=0.6).generate("goal", [], 1)
    assert (calls[0]["temperature"], calls[0]["max_tokens"]) == (0.6, 45)


def test_code_repair_retries_preserve_explicit_protocol(monkeypatch):
    client, calls = recording_client(temperature=0, max_tokens=77)
    monkeypatch.setattr(benchmark, "extract_visible_examples", lambda prompt: ["f()"])
    monkeypatch.setattr(benchmark, "_run_visible_doctest_against_code", lambda code, prompt: [{
        "args_repr": "f()", "got_repr": "1", "want_repr": "2", "doctest_line_repr": "f()",
    }])
    benchmark.generate_final_code(client, "def f(): pass", [])
    assert len(calls) == 3
    assert all(call["temperature"] == 0 and call["max_tokens"] == 77 for call in calls)


def test_private_key_selector_uses_only_requested_variable(tmp_path, monkeypatch):
    for name in ("PRIVATE_API_KEY", "PRIVATE_API_KEY_1", "VIB_LLM_MODEL", "VIB_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VIB_PRIVATE_API_KEY_ENV", "PRIVATE_API_KEY_1")
    env_path = tmp_path / "fake.env"
    env_path.write_text(
        'PRIVATE_API_URL="https://example.invalid/v1"\n'
        'PRIVATE_API_KEY="exhausted-test-key"\nPRIVATE_API_KEY_1="active-test-key"\n'
    )
    client = build_client("deepseek-v4-flash-0817", "private", env_path=str(env_path), log=False)
    assert client.api_key == "active-test-key"
    assert client.base_url == "https://example.invalid/v1"
    assert client.model == "deepseek-v4-flash-0817"


def test_missing_selected_key_never_falls_back_to_exhausted_key(tmp_path, monkeypatch):
    monkeypatch.delenv("PRIVATE_API_KEY_1", raising=False)
    monkeypatch.setenv("VIB_PRIVATE_API_KEY_ENV", "PRIVATE_API_KEY_1")
    env_path = tmp_path / "fake.env"
    env_path.write_text('PRIVATE_API_URL="https://example.invalid/v1"\nPRIVATE_API_KEY="old-test-key"\n')
    with pytest.raises(ValueError, match="PRIVATE_API_KEY_1"):
        build_client("deepseek-v4-flash-0817", "private", env_path=str(env_path), log=False)


def test_protocol_rejects_invalid_token_limit():
    from src.inference_protocol import InferenceProtocol

    with pytest.raises(ValueError, match="max_tokens"):
        InferenceProtocol(max_tokens=0)


def test_vanilla_uses_same_explicit_stage_protocol(monkeypatch):
    from src.inference_protocol import InferenceProtocol
    from experiments import run_vanilla_baseline as vanilla

    client, calls = recording_client()
    monkeypatch.setattr(vanilla.mb, "build_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(vanilla.mb.UnifiedEvaluator, "evaluate", lambda *args: True)
    protocol = InferenceProtocol(temperature=0, max_tokens=77)
    harness = vanilla.VanillaBaselineHarness(api_key="test-key", inference_protocol=protocol)
    sample = vanilla.mb.TaskSample(task_id="GSM8K/test", dataset_name="GSM8K",
                                  prompt="1+1", reference_answer="2")
    result = harness._run_vanilla_variant(sample)
    assert (calls[0]["temperature"], calls[0]["max_tokens"]) == (0, 77)
    assert result["inference_protocol"]["effective"] == protocol.describe()["effective"]
    assert result["token_stats"]["usage_complete"] is True
    assert len(result["request_records"]) == 1


def test_team_trace_protocol_matches_sent_parameters(monkeypatch):
    from src.inference_protocol import InferenceProtocol

    client, calls = recording_client()
    monkeypatch.setattr(benchmark, "build_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(benchmark.UnifiedEvaluator, "evaluate", lambda *args: True)

    class FakeWorkflow:
        def __init__(self, agents, **kwargs):
            self.agents = agents
            self.stats = {"total_approved": len(agents)}

        def run(self, goal, verbose=False):
            return {"history": [agent["generate"](goal, [], 1) for agent in self.agents]}

    monkeypatch.setattr(benchmark, "VIBMultiAgentWorkflow", FakeWorkflow)
    protocol = InferenceProtocol(temperature=0, max_tokens=77, math_max_tokens=99)
    harness = benchmark.MultiDatasetHarness(api_key="test-key", inference_protocol=protocol)
    sample = benchmark.TaskSample(task_id="GSM8K/test", dataset_name="GSM8K",
                                  prompt="1+1", reference_answer="2")
    result = harness._run_variant(sample, benchmark.create_math_agents,
                                  benchmark.generate_final_answer, benchmark.BypassVIBRouter())
    assert [call["max_tokens"] for call in calls] == [77, 77, 77, 99]
    assert all(call["temperature"] == 0 for call in calls)
    trace = harness._build_trace_payload(sample, "full", result)
    assert trace["inference_protocol"] == protocol.describe()
    assert [record["stage"] for record in trace["request_records"]] == ["agent"] * 3 + ["math"]


@pytest.mark.parametrize("mode", ["team", "vanilla"])
def test_cli_serializes_effective_stage_protocol_without_network(tmp_path, monkeypatch, mode):
    from experiments import run_vanilla_baseline as vanilla

    module = benchmark if mode == "team" else vanilla
    endpoint_module = benchmark if mode == "team" else vanilla.mb
    output = tmp_path / "result.json"
    captured = []

    class FakeHarness:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def run_benchmark(self, datasets, limit):
            return {"summary": {}, "inference_protocol": captured[0]["inference_protocol"].describe()}

    monkeypatch.setattr(endpoint_module, "build_client", lambda *args, **kwargs: SimpleNamespace(
        model="test-model", provider="private", api_key="fake-key",
    ))
    monkeypatch.setattr(module, "MultiDatasetHarness" if mode == "team" else "VanillaBaselineHarness", FakeHarness)
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(sys, "argv", [
        "runner", "--datasets", "gsm8k", "--output", str(output),
        "--temperature", "0", "--max-tokens", "77", "--math-max-tokens", "99",
    ])
    module.main()
    effective = json.loads(output.read_text())["inference_protocol"]["effective"]
    assert effective == {
        "agent": {"temperature": 0, "max_tokens": 77},
        "code": {"temperature": 0, "max_tokens": 77},
        "math": {"temperature": 0, "max_tokens": 99},
    }
    assert captured[0]["request_ledger_dir"] == str(output.with_suffix(".requests"))

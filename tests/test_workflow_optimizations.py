import os
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

from src.vib_router import VIBEvaluationResult, VIBSparseCommRouter
from src.workflow import VIBMultiAgentWorkflow
from experiments.run_multi_benchmark import MultiDatasetHarness, TaskSample, ExperimentConfig


class RecordingRouter:
    def __init__(self, allow_all=True):
        self.allow_all = allow_all
        self.history_lengths = []

    def evaluate_detailed(self, candidate_msg, history, goal, step_t, **kwargs):
        self.history_lengths.append(len(history))
        return VIBEvaluationResult(
            utility_score=1.0 if self.allow_all else -1.0,
            sufficiency=1.0,
            redundancy=0.0,
            beta_t=1.0,
            is_allowed=self.allow_all,
            threshold=0.0
        )


def make_agent(name, content):
    return {"name": name, "generate": lambda goal, history, step_t: content}


def test_frozen_same_round_history_does_not_chain_same_step_messages():
    router = RecordingRouter(allow_all=True)
    workflow = VIBMultiAgentWorkflow(
        agents=[
            make_agent("Planner", "Plan with numbers 1 + 2 = 3"),
            make_agent("Solver", "Solve with 3 * 4 = 12"),
        ],
        vib_router=router,
        max_steps=1,
        same_round_history_mode="frozen"
    )

    workflow.run("math goal", verbose=False)

    assert router.history_lengths == [0, 0]


def test_incremental_same_round_history_still_accumulates_within_step():
    router = RecordingRouter(allow_all=True)
    workflow = VIBMultiAgentWorkflow(
        agents=[
            make_agent("Planner", "Plan"),
            make_agent("Solver", "Solve"),
        ],
        vib_router=router,
        max_steps=1,
        same_round_history_mode="incremental"
    )

    workflow.run("math goal", verbose=False)

    assert router.history_lengths == [0, 1]


def test_reviewer_digest_fallback_keeps_edge_case_hint():
    router = RecordingRouter(allow_all=False)
    workflow = VIBMultiAgentWorkflow(
        agents=[make_agent("Reviewer", "This edge case fails when input is empty.")],
        vib_router=router,
        max_steps=1,
        enable_digest_fallback=True
    )

    result = workflow.run("code goal", verbose=False)

    assert workflow.stats["total_digests"] == 1
    assert any(msg.startswith("[Reviewer-digest]:") for msg in result["history"])


def test_math_role_beta_map_prioritizes_solver_and_verifier():
    router = VIBSparseCommRouter()

    planner_beta = router.compute_beta_t(step_t=1, role="Planner", goal="simple math question")
    solver_beta = router.compute_beta_t(step_t=1, role="Solver", goal="simple math question")
    verifier_beta = router.compute_beta_t(step_t=1, role="Verifier", goal="simple math question")

    assert solver_beta > verifier_beta > planner_beta


def test_math_role_weights_can_be_disabled():
    router = VIBSparseCommRouter(enable_math_role_weights=False)

    planner_beta = router.compute_beta_t(step_t=1, role="Planner", goal="simple math question")
    solver_beta = router.compute_beta_t(step_t=1, role="Solver", goal="simple math question")
    verifier_beta = router.compute_beta_t(step_t=1, role="Verifier", goal="simple math question")

    assert planner_beta == solver_beta == verifier_beta


def test_reasoning_guard_can_be_disabled():
    router = VIBSparseCommRouter(enable_reasoning_guard=False)

    assert router._detect_reasoning_signal("Let x = 3 + 4, therefore total is 7", role="Solver") is False


def test_harness_experiment_config_controls_gsm8k_workflow_options():
    harness = MultiDatasetHarness(
        api_key="dummy-key",
        experiment_config=ExperimentConfig(
            enable_gsm8k_frozen_history=False,
            enable_humaneval_digest=False
        )
    )

    gsm8k_sample = TaskSample(task_id="GSM8K/test", dataset_name="GSM8K", prompt="math")
    humaneval_sample = TaskSample(task_id="HumanEval/test", dataset_name="HumanEval", prompt="code")

    gsm8k_options = harness._workflow_options_for_sample(gsm8k_sample)
    humaneval_options = harness._workflow_options_for_sample(humaneval_sample)

    assert gsm8k_options["same_round_history_mode"] == "incremental"
    assert humaneval_options["enable_digest_fallback"] is False


def test_mbpp_early_stop_uses_probe_when_first_round_passes(monkeypatch):
    sample = TaskSample(
        task_id="MBPP/test",
        dataset_name="MBPP",
        prompt="implement function",
        entry_point="foo",
        test_cases=["assert foo() == 1"],
        reference_answer="def foo(): return 1"
    )
    harness = MultiDatasetHarness(api_key="dummy-key", workers=1)

    calls = []

    def fake_run_variant(sample, create_agents_fn, generate_fn, router, max_steps_override=None):
        calls.append(max_steps_override or harness.max_steps)
        return {
            "history": ["[Coder]: done"],
            "workflow_stats": {"total_approved": 1 if max_steps_override == 1 else 6},
            "output": "def foo(): return 1",
            "passed": True,
            "token_stats": {
                "total_calls": 1,
                "total_prompt_tokens": 10,
                "total_completion_tokens": 5,
                "total_tokens": 15
            },
            "early_stopped": False
        }

    monkeypatch.setattr(harness, "_select_task_components", lambda s: (None, None))
    monkeypatch.setattr(harness, "_run_variant", fake_run_variant)

    result = harness.run_single_task_with_trace(sample)

    assert calls == [1, 1]
    assert result["full"]["early_stopped"] is True
    assert result["vib"]["early_stopped"] is True
    assert result["task_result"].full_messages == 1
    assert result["task_result"].vib_messages == 1

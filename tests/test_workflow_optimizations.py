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


def make_counting_agent(name, content, counter):
    def generate(goal, history, step_t):
        counter[name] = counter.get(name, 0) + 1
        return content
    return {"name": name, "generate": generate}


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


def test_coder_digest_fallback_keeps_fix_hint():
    router = RecordingRouter(allow_all=False)
    workflow = VIBMultiAgentWorkflow(
        agents=[make_agent("Coder", "Fix the bug for empty input; otherwise this fails if the list is empty.")],
        vib_router=router,
        max_steps=1,
        enable_digest_fallback=True
    )

    result = workflow.run("code goal", verbose=False)

    assert workflow.stats["total_digests"] == 1
    assert any(msg.startswith("[Coder-digest]:") for msg in result["history"])


def test_first_round_role_guard_rescues_coder_and_reviewer():
    router = RecordingRouter(allow_all=False)
    workflow = VIBMultiAgentWorkflow(
        agents=[
            make_agent("Architect", "High-level plan."),
            make_agent("Coder", "Implement the main loop and handle edge constraints."),
            make_agent("Reviewer", "Check boundary cases and invalid input."),
        ],
        vib_router=router,
        max_steps=1,
        required_roles_first_step=["Coder", "Reviewer"]
    )

    result = workflow.run("code goal", verbose=False)

    assert workflow.stats["total_role_rescues"] == 2
    assert workflow.stats["total_approved"] == 2
    assert workflow.stats["total_blocked"] == 1
    assert result["history"] == [
        "[Coder]: Implement the main loop and handle edge constraints.",
        "[Reviewer]: Check boundary cases and invalid input."
    ]


def test_rescue_trace_marks_actual_candidate_not_first_rejection():
    workflow = VIBMultiAgentWorkflow(
        agents=[
            make_agent("Architect", "Plan"),
            make_agent("Coder", "Implement"),
            make_agent("Reviewer", "Review"),
        ],
        vib_router=RecordingRouter(allow_all=False),
        max_steps=1,
        required_roles_first_step=["Reviewer"],
    )
    workflow.run("goal", verbose=False)
    decisions = workflow.stats["gate_decisions"]
    assert [row["rescued"] for row in decisions] == [False, False, True]
    assert [row["is_allowed"] for row in decisions] == [False, False, True]


def test_first_round_role_guard_respects_minimum_approval_floor():
    router = RecordingRouter(allow_all=False)
    workflow = VIBMultiAgentWorkflow(
        agents=[
            make_agent("Architect", "High-level plan."),
            make_agent("Coder", "Implement core logic."),
            make_agent("Reviewer", "Check boundary conditions."),
        ],
        vib_router=router,
        max_steps=1,
        required_roles_first_step=["Coder", "Reviewer"],
        first_step_min_approvals=2
    )

    # Simulate the Architect already being allowed via digest fallback.
    workflow._should_emit_digest = lambda item, result: item["agent"] == "Architect"  # type: ignore[method-assign]
    workflow._build_digest_message = lambda item: f"[{item['agent']}-digest]: {item['content']}"  # type: ignore[method-assign]

    result = workflow.run("code goal", verbose=False)

    assert workflow.stats["total_role_rescues"] == 1
    assert len(result["history"]) == 2
    assert any(msg.startswith("[Architect-digest]:") for msg in result["history"])


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
            enable_humaneval_digest=False,
            enable_humaneval_first_round_role_guard=False
        )
    )

    gsm8k_sample = TaskSample(task_id="GSM8K/test", dataset_name="GSM8K", prompt="math")
    humaneval_sample = TaskSample(task_id="HumanEval/test", dataset_name="HumanEval", prompt="code")

    gsm8k_options = harness._workflow_options_for_sample(gsm8k_sample)
    humaneval_options = harness._workflow_options_for_sample(humaneval_sample)

    assert gsm8k_options["same_round_history_mode"] == "incremental"
    assert humaneval_options["enable_digest_fallback"] is False
    assert humaneval_options["required_roles_first_step"] == []


def test_harness_default_config_disables_humaneval_digest():
    harness = MultiDatasetHarness(api_key="dummy-key")
    sample = TaskSample(task_id="HumanEval/test", dataset_name="HumanEval", prompt="code")
    options = harness._workflow_options_for_sample(sample)

    assert harness.experiment_config.enable_humaneval_digest is False
    assert harness.experiment_config.enable_math_role_weights is False
    assert harness.experiment_config.enable_gsm8k_reasoning_guard is False
    assert harness.experiment_config.enable_gsm8k_frozen_history is True
    assert harness.experiment_config.enable_mbpp_early_stop is True
    assert options["enable_digest_fallback"] is False
    assert options["required_roles_first_step"] == ["Coder", "Reviewer"]
    assert options["first_step_min_approvals"] == 2
    gsm8k_sample = TaskSample(task_id="GSM8K/test", dataset_name="GSM8K", prompt="math")
    gsm8k_options = harness._workflow_options_for_sample(gsm8k_sample)
    assert gsm8k_sample.dataset_name == "GSM8K"
    assert gsm8k_options["same_round_history_mode"] == "frozen"


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


def test_workflow_without_scheduler_runs_every_agent_and_records_nothing():
    calls = {}
    workflow = VIBMultiAgentWorkflow(
        agents=[
            make_counting_agent("Planner", "Plan", calls),
            make_counting_agent("Solver", "Solve", calls),
        ],
        vib_router=RecordingRouter(allow_all=True),
        max_steps=1,
    )

    workflow.run("goal", verbose=False)

    assert calls == {"Planner": 1, "Solver": 1}
    assert workflow.stats["schedule_decisions"] == []


def test_pre_generation_scheduler_skips_unscheduled_agents():
    calls = {}
    workflow = VIBMultiAgentWorkflow(
        agents=[
            make_counting_agent("Planner", "Plan", calls),
            make_counting_agent("Solver", "Solve", calls),
        ],
        vib_router=RecordingRouter(allow_all=True),
        max_steps=1,
        pre_generation_scheduler=lambda goal, history, step_t, agent_names: ["Solver"],
    )

    workflow.run("goal", verbose=False)

    assert calls == {"Solver": 1}
    decision = workflow.stats["schedule_decisions"][0]
    assert decision["selected"] == ["Solver"]
    assert decision["skipped"] == ["Planner"]


def test_pre_generation_scheduler_receives_visible_evidence_only():
    seen = []

    def scheduler(goal, history, step_t, agent_names):
        seen.append({
            "goal": goal,
            "history": list(history),
            "step_t": step_t,
            "agents": list(agent_names),
        })
        return list(agent_names)

    workflow = VIBMultiAgentWorkflow(
        agents=[make_agent("Planner", "Plan")],
        vib_router=RecordingRouter(allow_all=True),
        max_steps=2,
        pre_generation_scheduler=scheduler,
    )

    workflow.run("the goal", verbose=False)

    assert seen[0] == {"goal": "the goal", "history": [], "step_t": 1, "agents": ["Planner"]}
    assert seen[1]["step_t"] == 2
    assert seen[1]["history"] == ["[Planner]: Plan"]


def test_pre_generation_scheduler_drops_unknown_names_and_can_skip_round():
    calls = {}
    workflow = VIBMultiAgentWorkflow(
        agents=[make_counting_agent("Solver", "Solve", calls)],
        vib_router=RecordingRouter(allow_all=True),
        max_steps=1,
        pre_generation_scheduler=lambda goal, history, step_t, agent_names: ["Ghost"],
    )

    workflow.run("goal", verbose=False)

    assert calls == {}
    assert workflow.stats["schedule_decisions"][0]["selected"] == []
    assert workflow.stats["total_candidates"] == 0

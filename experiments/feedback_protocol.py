"""Freeze public-only sampling, stage contracts and source identities."""

from decimal import Decimal
from importlib.metadata import version
import json
from pathlib import Path
import random

from experiments.calibrate_prompt_protocol import _bytes, _cost
from experiments.check_prompt_contrast_readiness import _load_draft
from experiments.prepare_gate_receiver import CONFIG, POLICIES, snapshot_evidence
from experiments.run_prompt_contrast_pilot import ROOT
from src.experiment_driver import file_sha256, implementation_sha256
from src.request_ledger import fingerprint


ROLES = {
    "Architect": "你是软件架构师。分析需求、边界条件与最佳算法策略。",
    "Coder": "你是核心程序员。根据需求编写高效 Python 代码。",
    "Reviewer": "你是代码审查员。严格检查边界漏洞与逻辑错误。",
}
PRIOR_FOLDERS = (
    "prompt_contrast_pilot_20261028", "prompt_contrast_replication_20261028",
    "gate_receiver_20261028",
)
SOURCE_FILES = (
    "experiments/feedback_protocol.py", "experiments/run_feedback_experiment.py",
    "experiments/analyze_feedback_experiment.py", "src/exploratory_prompt_session.py",
    "experiments/run_prompt_contrast_pilot.py", "experiments/calibrate_prompt_protocol.py",
    "experiments/check_prompt_contrast_readiness.py", "experiments/prepare_gate_receiver.py",
    "experiments/analyze_prompt_contrast_replication.py",
    "experiments/analyze_prompt_contrast_pilot.py", "src/llm_agents.py",
    "src/isolated_code_evaluation.py", "src/vib_router.py",
)


def agent_messages(goal, history, step, role):
    """Render the benchmark agent template without invoking its retrying client."""
    history_text = "\n".join(history) if history else "(暂无历史讨论)"
    user = f"""【全局任务目标】: {goal}

【当前公共讨论历史】:
{history_text}

【当前轮次】: 第 {step} 轮

请基于以上信息，以你的角色视角给出你的贡献。注意：
- 只输出你的发言内容（2-3句话）
- 不要输出角色名或前缀
- 提供新的、有价值的信息，避免重复已有内容"""
    return [{"role": "system", "content": ROLES[role]}, {"role": "user", "content": user}]


def freeze_feedback_protocol(endpoint):
    """Sample 16 remaining IDs without loading scores or reference code."""
    if (endpoint.get("provider") != "private" or not endpoint.get("credential_present")
            or endpoint.get("credential_env") != "PRIVATE_API_KEY_1"):
        raise ValueError("Explicit available second credential required")
    draft_path = ROOT / "results/prompt_contrast_preparation_20261028/draft.json"
    draft, digest = _load_draft(draft_path)
    excluded, priors = set(), []
    for folder, count in zip(PRIOR_FOLDERS, (24, 48, 32)):
        path = ROOT / "results" / folder / "manifest.json"
        prior = json.loads(path.read_text(encoding="utf-8"))
        ids = prior["task_ids"]
        if len(ids) != count or len(set(ids)) != count or excluded.intersection(ids):
            raise ValueError("Invalid or overlapping prior cohort")
        excluded.update(ids)
        priors.append({"path": str(path), "sha256": file_sha256(path)})
    streams = {row["task_id"]: row for row in draft["candidate_streams"]}
    if len(streams) != 164 or not excluded <= streams.keys():
        raise ValueError("Invalid public task pool")
    selected = random.Random(20261103).sample(sorted(streams.keys() - excluded), 16)
    proof = lambda item: {**item, "path": str(draft_path.parent / item["path"])}
    return {
        "version": "fresh_two_round_feedback_v1", "endpoint": endpoint,
        "task_ids": selected, "tasks": [{"task_id": task, "prompt": streams[task]["prompt"]}
                                       for task in selected],
        "excluded_task_ids": sorted(excluded), "priors": priors,
        "draft_path": str(draft_path), "draft_sha256": digest, "sample_seed": 20261103,
        "policies": list(POLICIES), "roles": ROLES, "router_config": CONFIG,
        "phase_counts": {"round1": 48, "round2": 192, "synthesis": 128},
        "phase_budgets": {"round1": "0.15", "round2": "0.45", "synthesis": "0.40"},
        "parameters": {"agent": {"temperature": 0, "max_tokens": 256},
                       "code": {"temperature": 0, "max_tokens": 1024}},
        "encoder": {**snapshot_evidence(), "dimension": 384, "max_seq_length": 256,
                    "device": "cpu", "eval_mode": True, "dtype": "float32_l2_normalized",
                    "versions": {package: version(package) for package in (
                        "sentence-transformers", "torch", "transformers", "numpy")}},
        "evaluation": proof(draft["evaluation"]),
        "prompt_source": proof(draft["prompt_sources"]["specification_priority"]),
        "analysis": {"unit": "task_mean_over_two_repetitions", "primary": ["vib", "random_matched"],
                     "primary_alpha": 0.05, "bootstrap_samples": 10000, "seed": 20261103,
                     "secondary": "exploratory_unadjusted"},
        "scope": "two_round_feedback_shared_round1_single_team_trajectory_conditional_count_matching",
        "candidate_truncation": "retain_nonempty_paid_text_no_repair",
        "final_truncation": "failure_no_regeneration",
        "matching": "each_round_k_from_vib_own_candidates_controls_select_own_candidates",
        "same_round_history": "incremental_gate_all_agents_see_round_start_history",
        "price_scope": "historical_proxy_not_actual_invoice",
        "price_date": "2026-10-23", "input_usd_per_million": "0.30",
        "output_usd_per_million": "1.20", "total_proxy_stop_usd": "1",
        "budget_scope": "observed_proxy_fail_fast_no_invoice_or_verified_token_bound",
        "sdk_retries": 0, "attempts_per_job": 1, "workers": 1, "timeout_seconds": 60,
        "request_implementation_sha256": implementation_sha256(),
        "sources_sha256": {name: file_sha256(ROOT / name) for name in SOURCE_FILES},
    }


def validate_protocol(protocol):
    if protocol != freeze_feedback_protocol(protocol["endpoint"]):
        raise ValueError("Frozen feedback protocol inputs or implementation changed")


def stage_manifest(protocol, phase, parents, routes):
    """Freeze every request in a phase before sending any of that phase."""
    if phase not in protocol["phase_counts"]:
        raise ValueError("Unknown feedback phase")
    if phase == "round1" and (parents or routes is not None):
        raise ValueError("Round one has no parent history")
    if phase != "round1" and (not parents or routes is None):
        raise ValueError("Downstream phase requires frozen parent evidence")
    jobs = []
    renderer = None
    if phase == "synthesis":
        proof = protocol["prompt_source"]
        if file_sha256(proof["path"]) != proof["sha256"]:
            raise ValueError("Frozen synthesis renderer changed")
        namespace = {}
        exec(compile(Path(proof["path"]).read_text(encoding="utf-8"), proof["path"], "exec"), namespace)
        renderer = namespace["specification_priority"]
    for task in protocol["tasks"]:
        task_id, goal = task["task_id"], task["prompt"]
        for policy in (("shared",) if phase == "round1" else protocol["policies"]):
            history = [] if routes is None else routes["histories"][task_id][policy]["history"]
            items = (0, 1) if renderer else tuple(ROLES)
            for item in items:
                stage = "code" if renderer else "agent"
                messages = ([{"role": "user", "content": renderer(goal, history)}] if renderer
                            else agent_messages(goal, history, 1 if phase == "round1" else 2, item))
                jobs.append({
                    "job_id": f"{task_id}/{policy}/{item}", "task_id": task_id,
                    "prompt_variant": policy, "repetition": item if renderer else 0,
                    "agent_role": None if renderer else item, "stage": stage,
                    "messages": messages, "messages_sha256": fingerprint(messages),
                    "history_sha256": fingerprint(history), "parameters": protocol["parameters"][stage],
                    "planning_input_units": _bytes(messages) + 1024,
                })
    random.Random(20261103 + ("round1", "round2", "synthesis").index(phase)).shuffle(jobs)
    scenario = sum(_cost(job["planning_input_units"], job["parameters"]["max_tokens"]) for job in jobs)
    if len(jobs) != protocol["phase_counts"][phase] or scenario > Decimal(protocol["phase_budgets"][phase]):
        raise ValueError("Complete phase planning scenario exceeds its nontransferable budget")
    return {
        "version": "feedback_phase_v1", "phase": phase, "protocol_sha256": fingerprint(protocol),
        "endpoint": protocol["endpoint"], "parents": parents, "routes": routes,
        "jobs": jobs, "proxy_stop_usd": protocol["phase_budgets"][phase],
        "planned_proxy_scenario_usd": str(scenario), "price_scope": protocol["price_scope"],
    }

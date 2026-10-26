"""核验完整配对覆盖，并按任务聚类分析共同候选流探索结果。"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_frozen_replay import BOUNDARY_POLICIES, POLICIES, aggregate, fingerprint, write_json

CAVEATS = {
    "math": [
        "复用历史候选，非独立泛化评测；不覆盖非整数答案。",
        "重复合成不是新增独立问题；所有区间未经多重比较校正。",
        "费用只覆盖当前合成，不含历史候选生成，不能当作端到端调度成本。",
        "VIB与截轮提示词相同时，结果差异不能归因于选择机制。",
        "direct不是等预算单智能体对照。",
        "随机放行率与余弦阈值固定，未在该任务子集校准为等消息预算。",
    ],
    "humaneval": [
        "复用历史候选，非独立泛化评测；只覆盖两轮六候选齐全的 HumanEval 任务。",
        "重复合成不是新增独立问题；所有区间未经多重比较校正。",
        "费用只覆盖当前合成，不含历史候选生成，不能当作端到端调度成本。",
        "合成走单次调用；主 harness 另有只看题面示例的 doctest 自修复回路，未纳入对照。",
        "direct不是等预算单智能体对照。",
        "随机放行率与余弦阈值固定，未在该任务子集校准为等消息预算。",
    ],
}

BOUNDARY_CONTRASTS = (
    ("full", "first_round"),
    ("full", "direct"),
    ("first_round", "direct"),
    ("vib", "first_round"),
    ("vib", "random"),
    ("vib", "cosine"),
)
CODE_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
WORDS = re.compile(r"\w+")


def history_features(messages):
    """描述可见候选；不读参考答案，不把代码结构差异当成语义冲突。"""
    if len(messages) != 6:
        raise ValueError("边界审计仅支持两轮六候选")
    bodies = [message.partition("]: ")[2] or message for message in messages]
    first_text = "\n".join(bodies[:3])
    second_text = "\n".join(bodies[3:])
    first_words = set(WORDS.findall(first_text.lower()))
    second_words = set(WORDS.findall(second_text.lower()))
    code_structures = set()
    parsed_blocks = 0
    total_blocks = 0
    for body in bodies:
        for block in CODE_FENCE.findall(body):
            total_blocks += 1
            try:
                structure = ast.dump(ast.parse(block.strip()), include_attributes=False)
            except (SyntaxError, ValueError):
                continue
            parsed_blocks += 1
            code_structures.add(hashlib.sha256(structure.encode()).hexdigest())
    total_chars = len(first_text) + len(second_text)
    return {
        "candidate_messages": len(messages),
        "first_round_chars": len(first_text),
        "second_round_chars": len(second_text),
        "second_round_char_fraction": len(second_text) / total_chars if total_chars else 0.0,
        "second_round_new_word_fraction": (
            len(second_words - first_words) / len(second_words) if second_words else 0.0
        ),
        "python_fence_blocks": total_blocks,
        "parsable_python_blocks": parsed_blocks,
        "distinct_python_ast_hashes": sorted(code_structures),
        "distinct_python_ast_count": len(code_structures),
    }


def compare_policies(task_ids, repeats, lookup, policy, baseline):
    """返回任务聚类区间与观测差异；相同 repeat 编号不代表共享随机数。"""
    tasks = []
    for task_id in task_ids:
        outcomes = Counter()
        policy_passes = baseline_passes = identical = identical_disagreements = 0
        for repeat in range(repeats):
            selected = lookup[task_id, repeat, policy]
            control = lookup[task_id, repeat, baseline]
            selected_passed = bool(selected["passed"])
            control_passed = bool(control["passed"])
            policy_passes += selected_passed
            baseline_passes += control_passed
            if selected_passed == control_passed:
                outcomes["both_pass" if selected_passed else "both_fail"] += 1
            else:
                outcomes["policy_only_pass" if selected_passed else "baseline_only_pass"] += 1
            if selected["prompt_sha256"] == control["prompt_sha256"]:
                identical += 1
                identical_disagreements += selected_passed != control_passed
        difference = (policy_passes - baseline_passes) / repeats
        tasks.append({
            "task_id": task_id,
            "policy_pass_rate": policy_passes / repeats,
            "baseline_pass_rate": baseline_passes / repeats,
            "delta_pp": 100 * difference,
            "observed_direction": (
                "better" if difference > 0 else "worse" if difference < 0 else "tie"
            ),
            "repeat_outcomes": {
                key: outcomes[key] for key in (
                    "both_pass", "both_fail", "policy_only_pass", "baseline_only_pass"
                )
            },
            "opposite_repeat_differences": (
                outcomes["policy_only_pass"] > 0 and outcomes["baseline_only_pass"] > 0
            ),
            "identical_prompt_pairs": identical,
            "identical_prompt_score_disagreements": identical_disagreements,
        })
    summary = {
        **paired_interval([task["delta_pp"] / 100 for task in tasks]),
        "tasks_better": sum(task["observed_direction"] == "better" for task in tasks),
        "tasks_worse": sum(task["observed_direction"] == "worse" for task in tasks),
        "tasks_tied": sum(task["observed_direction"] == "tie" for task in tasks),
        "tasks_with_opposite_repeat_differences": sum(
            task["opposite_repeat_differences"] for task in tasks
        ),
        "identical_prompt_pairs": sum(task["identical_prompt_pairs"] for task in tasks),
        "identical_prompt_score_disagreements": sum(
            task["identical_prompt_score_disagreements"] for task in tasks
        ),
        "tasks": tasks,
    }
    return summary


def paired_interval(differences, seed=20261023):
    """每题先聚合重复合成，再重采样任务；区间仅用于探索。"""
    values = np.asarray(differences, dtype=float)
    if not len(values):
        raise ValueError("空配对任务集")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values))).mean(axis=1)
    return {
        "delta_pp": float(100 * values.mean()),
        "bootstrap_ci95_pp": [float(100 * value) for value in np.quantile(samples, [0.025, 0.975])],
        "independent_tasks": len(values),
    }


def analyze(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("version") not in (2, 3) or manifest.get("router_config", {}).get("beta_0") != 3.2:
        raise ValueError("路由协议未对齐论文配置，禁止纳入本轮分析")
    is_boundary = manifest["version"] == 3
    policies = BOUNDARY_POLICIES if is_boundary else POLICIES
    if tuple(manifest.get("policies", ())) != policies:
        raise ValueError("策略列表与协议不一致")
    if is_boundary:
        if manifest.get("protocol") != "boundary" or not manifest.get("exclusions"):
            raise ValueError("边界协议缺少旧题排除记录")
        if manifest.get("evaluation", {}).get("independent_code_checks") != 2:
            raise ValueError("边界协议缺少独立评分检查")
        excluded = {
            task_id for item in manifest["exclusions"] for task_id in item["task_ids"]
        }
        if excluded & {task["task_id"] for task in manifest["tasks"]}:
            raise ValueError("边界任务与旧 pilot 重叠")
    dataset = manifest.get("dataset", "math")
    if dataset not in CAVEATS:
        raise ValueError(f"未知数据集 {dataset!r}，拒绝分析")
    records = [json.loads(path.read_text()) for path in directory.glob("request_*.json")]
    task_ids = [task["task_id"] for task in manifest["tasks"]]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("重复任务")
    expected = {
        (task_id, repeat, policy)
        for task_id in task_ids for repeat in range(manifest["repeats"])
        for policy in policies
    }
    observed = [(row["task_id"], row["repeat"], row["policy"]) for row in records]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("重复请求或未完整配对，禁止分析中途准确率")
    if any(row["status"] != "ok" for row in records):
        raise ValueError("存在无效请求")
    contract_hash = fingerprint(manifest)
    for row in records:
        expected_id = fingerprint([
            contract_hash, row["task_id"], row["repeat"], row["policy"]
        ])
        if row.get("request_id") != expected_id:
            raise ValueError("请求身份与冻结契约不匹配")
        if is_boundary and row.get("evaluation_seed") != manifest["evaluation"]["python_random_seed"]:
            raise ValueError("评分种子与边界契约不一致")
        if is_boundary and row.get("response_model") != manifest["model"]:
            raise ValueError("返回模型与边界契约不一致")
    lookup = {(row["task_id"], row["repeat"], row["policy"]): row for row in records}
    scores = {
        policy: {
            task_id: np.mean([
                lookup[task_id, repeat, policy]["passed"]
                for repeat in range(manifest["repeats"])
            ])
            for task_id in task_ids
        } for policy in policies
    }
    comparisons = {}
    for policy in policies:
        if policy == "full":
            continue
        differences = [scores[policy][task_id] - scores["full"][task_id] for task_id in task_ids]
        comparisons[f"{policy}_minus_full"] = {
            **paired_interval(differences),
            "tasks_better": sum(bool(value > 0) for value in differences),
            "tasks_worse": sum(bool(value < 0) for value in differences),
        }
    identical = 0 if is_boundary else sum(
        lookup[task_id, repeat, "vib"]["prompt_sha256"]
        == lookup[task_id, repeat, "first_round"]["prompt_sha256"]
        for task_id in task_ids for repeat in range(manifest["repeats"])
    )
    if is_boundary and any(
        lookup[task_id, repeat, "first_round"]["prompt_sha256"]
        != lookup[task_id, repeat, "first_round_control"]["prompt_sha256"]
        for task_id in task_ids for repeat in range(manifest["repeats"])
    ):
        raise ValueError("首轮重复对照提示词不相同")
    costs = aggregate(records, policies)
    baseline = costs["full"]["prompt_tokens"] + costs["full"]["completion_tokens"]
    for policy in policies:
        tokens = costs[policy]["prompt_tokens"] + costs[policy]["completion_tokens"]
        costs[policy]["synthesis_token_saving_pct"] = 100 * (1 - tokens / baseline)
    boundary = {
        "contrasts": {
            f"{policy}_minus_{baseline}": compare_policies(
                task_ids, manifest["repeats"], lookup, policy, baseline
            ) for policy, baseline in (
                BOUNDARY_CONTRASTS if not is_boundary else (
                    ("full", "first_round"), ("full", "direct"),
                    ("first_round", "direct"), ("first_round_control", "first_round"),
                )
            )
        },
        "task_features": [
            {"task_id": task["task_id"], **history_features(task["candidates"])}
            for task in manifest["tasks"]
        ],
        "limitations": [
            "逐题标签是已观测合成结果之差，不是单条消息的因果效应或未来可预测真值。",
            "repeat 编号只作配对索引；API 未提供共享随机种子，不能排除合成或评分波动。",
            "提示词相同却评分不同的配对不能归因于门控；任务平均相等也不代表重复结果完全一致。",
            "可见消息特征只供描述，不训练选择器、不扫描阈值，不证明其能预测收益。",
            "Python AST 差异是语法结构差异，不证明语义冲突或错误；未执行候选代码。",
            "第二轮特征只有第二轮已生成后才可见，不能当作生成前免费调度信号。",
            "两批数据集分别分析；不合并准确率、费用或独立任务数为总体收益。",
        ],
    }
    return {
        "contract_sha256": contract_hash,
        "valid_requests": len(records), "independent_tasks": len(task_ids),
        "proxy_usd": sum(row["charged_proxy_usd"] for row in records),
        "policies": costs, "paired_comparisons": comparisons,
        "vib_first_round_identical_prompts": identical,
        "vib_first_round_total_pairs": 0 if is_boundary else len(task_ids) * manifest["repeats"],
        "scope": f"exploratory_{dataset}_subset_receiver_only",
        "caveats": CAVEATS[dataset] if not is_boundary else [
            "复用历史候选的新子集，排除旧 pilot 题号并不意味着 truly held-out。",
            "重复合成按任务聚类，主比较已冻结，但区间仍是探索性、未经多重比较校正。",
            "费用仅覆盖当前合成，不含候选生成；direct 不是等预算单智能体对照。",
            "首轮重复对照使用独立调用，没有共享 API 随机种子，不与首轮池化增大分母。",
            "代码评分固定 Python random 种子并独立执行两次，未控制其它随机源。",
            "新批改变了评分随机种子协议，不直接把新旧子集差异归因于任务迁移。",
        ],
        "collaboration_boundary": boundary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = analyze(args.directory)
    write_json(args.directory / "analysis.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

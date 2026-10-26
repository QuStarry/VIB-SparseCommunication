"""核验完整配对覆盖，并按任务聚类分析共同候选流探索结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_frozen_replay import POLICIES, aggregate, fingerprint, write_json

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
    if manifest.get("version") != 2 or manifest.get("router_config", {}).get("beta_0") != 3.2:
        raise ValueError("路由协议未对齐论文配置，禁止纳入本轮分析")
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
        for policy in POLICIES
    }
    observed = [(row["task_id"], row["repeat"], row["policy"]) for row in records]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("重复请求或未完整配对，禁止分析中途准确率")
    if any(row["status"] != "ok" for row in records):
        raise ValueError("存在无效请求")
    lookup = {(row["task_id"], row["repeat"], row["policy"]): row for row in records}
    scores = {
        policy: {
            task_id: np.mean([
                lookup[task_id, repeat, policy]["passed"]
                for repeat in range(manifest["repeats"])
            ])
            for task_id in task_ids
        } for policy in POLICIES
    }
    comparisons = {}
    for policy in POLICIES:
        if policy == "full":
            continue
        differences = [scores[policy][task_id] - scores["full"][task_id] for task_id in task_ids]
        comparisons[f"{policy}_minus_full"] = {
            **paired_interval(differences),
            "tasks_better": sum(bool(value > 0) for value in differences),
            "tasks_worse": sum(bool(value < 0) for value in differences),
        }
    identical = sum(
        lookup[task_id, repeat, "vib"]["prompt_sha256"]
        == lookup[task_id, repeat, "first_round"]["prompt_sha256"]
        for task_id in task_ids for repeat in range(manifest["repeats"])
    )
    costs = aggregate(records)
    baseline = costs["full"]["prompt_tokens"] + costs["full"]["completion_tokens"]
    for policy in POLICIES:
        tokens = costs[policy]["prompt_tokens"] + costs[policy]["completion_tokens"]
        costs[policy]["synthesis_token_saving_pct"] = 100 * (1 - tokens / baseline)
    return {
        "contract_sha256": fingerprint(manifest),
        "valid_requests": len(records), "independent_tasks": len(task_ids),
        "proxy_usd": sum(row["charged_proxy_usd"] for row in records),
        "policies": costs, "paired_comparisons": comparisons,
        "vib_first_round_identical_prompts": identical,
        "vib_first_round_total_pairs": len(task_ids) * manifest["repeats"],
        "scope": f"exploratory_{dataset}_subset_receiver_only",
        "caveats": CAVEATS[dataset],
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

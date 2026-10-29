"""Post-generation paired analysis, with independent tasks as bootstrap units."""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import random

from experiments.calibrate_prompt_protocol import _cost
from experiments.run_prompt_contrast_pilot import (
    freeze_pilot, response_path, validate_generation, write_json,
)
from src.experiment_driver import file_sha256
from src.isolated_code_evaluation import load_evaluators
from src.request_ledger import fingerprint


def paired_statistics(rows, *, samples=10000, seed=20261028):
    """Average repetitions first; bootstrap whole tasks, never select best output."""
    if not rows or isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("Nonempty paired data and positive bootstrap count required")
    variants = sorted({row["prompt_variant"] for row in rows})
    tasks = sorted({row["task_id"] for row in rows})
    indexed = {}
    for row in rows:
        key = (row["task_id"], row["prompt_variant"], row["repetition"])
        if key in indexed or row["repetition"] not in (0, 1) or type(row["passed"]) is not bool:
            raise ValueError("Invalid or duplicate scoring row")
        indexed[key] = int(row["passed"])
    expected = {(task, variant, repeat)
                for task in tasks for variant in variants for repeat in (0, 1)}
    if set(indexed) != expected or len(variants) < 2:
        raise ValueError("Incomplete paired repetitions")
    means = {
        variant: [(indexed[(task, variant, 0)] + indexed[(task, variant, 1)]) / 2
                  for task in tasks] for variant in variants}
    groups = {
        variant: {"passed_generations": sum(indexed[(task, variant, repeat)]
                                            for task in tasks for repeat in (0, 1)),
                  "generations": len(tasks) * 2, "independent_tasks": len(tasks),
                  "task_mean_accuracy_pct": 100 * sum(means[variant]) / len(tasks)}
        for variant in variants}
    comparisons = []
    for left, right in combinations(variants, 2):
        differences = [a - b for a, b in zip(means[left], means[right])]
        rng = random.Random(seed)
        bootstraps = sorted(
            100 * sum(differences[rng.randrange(len(tasks))] for _ in tasks) / len(tasks)
            for _ in range(samples))
        comparisons.append({
            "left": left, "right": right, "delta_pp": 100 * sum(differences) / len(tasks),
            "ci95_pp": [bootstraps[int(0.025 * (samples - 1))],
                        bootstraps[int(0.975 * (samples - 1))]],
            "independent_tasks": len(tasks),
            "left_higher_tasks": sum(value > 0 for value in differences),
            "right_higher_tasks": sum(value < 0 for value in differences),
            "tied_tasks": sum(value == 0 for value in differences),
        })
    return {"groups": groups, "comparisons": comparisons,
            "bootstrap_samples": samples, "seed": seed,
            "unit": "task_mean_over_two_repetitions",
            "interval_method": "empirical_percentile_order_statistics",
            "interpretation": "exploratory_unadjusted_not_equivalence_or_confirmation"}


def _readme(output, result):
    groups = result["statistics"]["groups"]
    costs = result["costs"]
    text = [
        "# 24题真实提示词配对小试验", "",
        "固定抽样24题，共144次生成，三种提示各两次。生成阶段不读取隐藏测试，"
        "不按测试补生成或选优；同题共用历史Full六候选，温度0、输出上限1024、无自动重试。",
        "", "## 真实结果", "",
        "| 提示 | 成功生成 | 任务均值准确率 | 输入Token | 输出Token | 历史代理费用 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    names = {"legacy_compact": "历史简洁", "current_arbitration": "当前仲裁",
             "specification_priority": "规格优先"}
    for variant, group in groups.items():
        cost = costs[variant]
        text.append(f'| {names[variant]} | {group["passed_generations"]}/{group["generations"]} | '
                    f'{group["task_mean_accuracy_pct"]:.2f}% | {cost["prompt_tokens"]} | '
                    f'{cost["completion_tokens"]} | ${cost["proxy_usd"]} |')
    text.extend([
        "", "准确率是全部预声明生成的成功比例；统计独立单位仍为24题，"
        "每题先取两次结果均值。不是best-of-two。截断直接计失败并保留费用。",
        "", "## 配对差异", "",
        "| 左组减右组 | 差值（百分点） | 任务级bootstrap 95%区间 |",
        "|---|---:|---|",
    ])
    for comparison in result["statistics"]["comparisons"]:
        lower, upper = comparison["ci95_pp"]
        text.append(f'| {names[comparison["left"]]} - {names[comparison["right"]]} | '
                    f'{comparison["delta_pp"]:+.2f} | [{lower:+.2f}, {upper:+.2f}] |')
    summary = result["generation_summary"]
    text.extend([
        "", "## 费用与边界", "",
        f'真实合计{summary["total_tokens"]} Token，按2026-10-23历史代理价计'
        f'${summary["proxy_cost_usd"]}；$1为观察停机阈值。'
        "未取得网关实际美元或人民币账单，不保证invoice硬上限，也未证明提供方Token上界。",
        f'截断{summary["truncated_responses"]}次，用量完整：{summary["usage_complete"]}。'
        "请求模型匹配仅证明本轮返回声明，非官方不可变快照验证。",
        "", "这是已审计存档上的探索性prompt-only对照，不是新独立确认集，"
        "不是VIB/Full因果对照，也非等实际费用实验。小样本及未校正多重比较限制推断；"
        "区间跨零不代表等效。没有修改历史主结果或论文主表。",
        "", "完整输出做两次独立解释器评分，最小环境不继承凭证。"
        "这是进程与状态隔离，不是恶意代码的网络/文件系统安全沙箱。",
        "", "## 工作流", "", "```mermaid", "flowchart LR",
        "A[固定抽24题和144请求] --> B[保存原文与真实usage]",
        "B --> C[生成全部完成后加载隐藏评分]", "C --> D[两次独立进程检查]",
        "D --> E[先聚合同题重复再配对分析]", "```",
        "", "## 一键导航", "",
        "| 内容 | 入口 |", "|---|---|",
        "| 冻结协议和请求顺序 | [manifest.json](manifest.json) |",
        "| 实际用量与请求身份 | [generation_report.json](generation_report.json) |",
        "| 发送前pending及终态事件 | [requests.jsonl](requests.jsonl) |",
        "| 逐题评分与配对区间 | [analysis.json](analysis.json) |",
        "| 完整响应原文 | [responses/](responses/) |", "",
    ])
    path = output / "README.md"
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(text))
            stream.flush()
            os.fsync(stream.fileno())


def analyze_pilot(output_dir):
    """No transport or generation adapter is invoked during scoring or recovery."""
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest != freeze_pilot(manifest["draft_path"], manifest["endpoint"])
            or manifest["analysis_implementation_sha256"] != file_sha256(__file__)):
        raise ValueError("Frozen generation or analysis implementation differs")
    generation_path = output / "generation_report.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    records = validate_generation(manifest, output, generation)
    if not generation["summary"]["run_complete"] or not generation["summary"]["usage_complete"]:
        raise ValueError("Incomplete generation cannot produce paired accuracy")
    analysis_path = output / "analysis.json"
    identity = {"manifest_sha256": fingerprint(manifest),
                "generation_report_sha256": file_sha256(generation_path),
                "implementation_sha256": file_sha256(__file__)}
    if analysis_path.exists():
        cached = json.loads(analysis_path.read_text(encoding="utf-8"))
        if any(cached.get(key) != value for key, value in identity.items()):
            raise ValueError("Cached analysis provenance differs")
        _readme(output, cached)
        return cached
    evaluators = load_evaluators(manifest["evaluation"]["path"], manifest["evaluation"]["sha256"])
    by_operation = {record["operation_id"]: record for record in records}
    evaluations, costs = [], {}
    for job in manifest["jobs"]:
        record = by_operation[fingerprint([fingerprint(manifest), job["job_id"]])]
        raw = json.loads(response_path(output, job).read_text(encoding="utf-8"))
        passed = False if record["status"] == "truncated" else evaluators[job["task_id"]](raw["content"])
        if type(passed) is not bool:
            raise ValueError("Scorer must return a boolean")
        evaluations.append({key: job[key] for key in ("job_id", "task_id", "prompt_variant", "repetition")}
                           | {"passed": passed, "status": record["status"],
                              "output_sha256": raw["output_sha256"]})
        cost = costs.setdefault(job["prompt_variant"], {"prompt_tokens": 0, "completion_tokens": 0})
        for field in ("prompt_tokens", "completion_tokens"):
            cost[field] += record["usage"][field]
    for cost in costs.values():
        cost["proxy_usd"] = str(_cost(cost["prompt_tokens"], cost["completion_tokens"]))
    result = {**identity, "generation_summary": generation["summary"], "evaluations": evaluations,
              "costs": costs, "statistics": paired_statistics(
                  evaluations, samples=manifest["analysis"]["bootstrap_samples"],
                  seed=manifest["analysis"]["seed"])}
    # Recheck artifacts after scoring, before publishing a complete result.
    validate_generation(manifest, output, generation)
    if identity["generation_report_sha256"] != file_sha256(generation_path):
        raise ValueError("Generation report changed during scoring")
    write_json(analysis_path, result)
    _readme(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_pilot(args.output_dir)
    print(json.dumps({"statistics": result["statistics"], "costs": result["costs"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

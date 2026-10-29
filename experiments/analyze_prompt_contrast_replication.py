"""Analyze disjoint replication separately, using the prospectively fixed contrast."""

import argparse
import json
import math
import os
from pathlib import Path

from experiments.analyze_prompt_contrast_pilot import paired_statistics
from experiments.calibrate_prompt_protocol import _cost
from experiments.run_prompt_contrast_pilot import response_path, write_json
from experiments.run_prompt_contrast_replication import validate_replication
from src.experiment_driver import file_sha256
from src.exploratory_prompt_session import validate_exploratory_generation
from src.isolated_code_evaluation import load_evaluators
from src.request_ledger import fingerprint


NAMES = {"legacy_compact": "历史简洁", "current_arbitration": "当前仲裁",
         "specification_priority": "规格优先"}


def exact_sign_test(left_higher, right_higher):
    """Two-sided exact binomial direction test, removing task-mean ties."""
    if any(type(value) is not int or value < 0 for value in (left_higher, right_higher)):
        raise ValueError("Nonnegative integer task counts required")
    count = left_higher + right_higher
    tail = sum(math.comb(count, index) for index in range(min(left_higher, right_higher) + 1))
    return min(1.0, 2 * tail / 2 ** count)


def replication_statistics(rows, settings):
    stats = paired_statistics(rows, samples=settings["bootstrap_samples"], seed=settings["seed"])
    for comparison in stats["comparisons"]:
        comparison["exact_sign_p"] = exact_sign_test(
            comparison["left_higher_tasks"], comparison["right_higher_tasks"])
    left, right = settings["primary"]
    source = next(row for row in stats["comparisons"] if {row["left"], row["right"]} == {left, right})
    primary = dict(source)
    if primary["left"] != left:
        primary.update(left=left, right=right, delta_pp=-source["delta_pp"],
                       ci95_pp=[-source["ci95_pp"][1], -source["ci95_pp"][0]],
                       left_higher_tasks=source["right_higher_tasks"],
                       right_higher_tasks=source["left_higher_tasks"])
    primary["alpha"] = settings["primary_alpha"]
    primary["role"] = "prospectively_fixed_primary_contrast"
    return stats, primary


def _readme(output, result):
    primary = result["primary_comparison"]
    summary = result["generation_summary"]
    lower, upper = primary["ci95_pp"]
    text = [
        "# 48题不重叠提示词复验", "",
        "从首轮未抽中的140题中，以seed20261030固定抽48题。三提示各题两次，"
        "共288个预声明请求；不修改提示词、温度0、输出cap1024，单次串行尝试。",
        "", "先完成全部生成，再载入隐藏评分。每份完整输出两次独立进程检查，"
        "截断按失败计账不补生成；同题两次先取均值，不是best-of-two。", "",
        "## 本轮单独结果", "",
        "| 提示 | 成功/生成 | 48题均值准确率 | 输入Token | 输出Token | 历史代理费用 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, group in result["statistics"]["groups"].items():
        cost = result["costs"][variant]
        text.append(f'| {NAMES[variant]} | {group["passed_generations"]}/{group["generations"]} | '
                    f'{group["task_mean_accuracy_pct"]:.2f}% | {cost["prompt_tokens"]} | '
                    f'{cost["completion_tokens"]} | ${cost["proxy_usd"]} |')
    text.extend([
        "", "## 预先固定主比较", "",
        f'历史简洁减当前仲裁：{primary["delta_pp"]:+.2f}个百分点，'
        f'10000次任务级bootstrap 95%区间[{lower:+.2f}, {upper:+.2f}]。',
        f'任务均值方向：历史简洁更高{primary["left_higher_tasks"]}题，'
        f'当前仲裁更高{primary["right_higher_tasks"]}题，平局{primary["tied_tasks"]}题；'
        f'双侧精确符号检验p={primary["exact_sign_p"]:.6f}，预声明alpha=0.05。',
        "", "bootstrap估计平均成功率差异；符号检验仅检验非平局题的方向，"
        "两者不是同一个统计量。不能只挑有利统计量，区间跨零不代表等效。",
        "", "## 探索性其他比较", "",
        "| 左组减右组 | 差值pp | bootstrap 95%区间 | 精确符号p |",
        "|---|---:|---|---:|",
    ])
    for comparison in result["statistics"]["comparisons"]:
        if {comparison["left"], comparison["right"]} == {"legacy_compact", "current_arbitration"}:
            continue
        lo, hi = comparison["ci95_pp"]
        text.append(f'| {NAMES[comparison["left"]]} - {NAMES[comparison["right"]]} | '
                    f'{comparison["delta_pp"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | '
                    f'{comparison["exact_sign_p"]:.6f} |')
    text.extend([
        "", "另外两对未校正多重比较，不按它们替换主比较。", "",
        "## 费用与推断边界", "",
        f'真实输入{summary["prompt_tokens"]}、输出{summary["completion_tokens"]}、'
        f'合计{summary["total_tokens"]} Token；历史代理费用${summary["proxy_cost_usd"]}。'
        f'截断{summary["truncated_responses"]}次，usage完整：{summary["usage_complete"]}。',
        "$1是历史代理价观察停机阈值，未取得网关实际美元/人民币账单，"
        "字节规划场景不是verified Token上界，不保证invoice硬上限。",
        "", "48题与首轮24题ID无交集，但都来自此前已审计的同一HumanEval存档，"
        "不是全新独立数据集，也不能排除预训练污染。模型ID匹配不是官方immutable snapshot认证。"
        "提示长度不同，不是等实际费用实验；只使用Full共同候选流，不证明VIB/Full因果优势。",
        "", "首轮与复验单独报告，不合并成确认性72题结果。"
        "暂不改变核心默认或论文主表，不按评分补生成。", "",
        "隔离评分不继承凭证，但不是恶意代码的网络/文件系统安全沙箱。", "",
        "## 工作流", "", "```mermaid", "flowchart LR",
        "A[排除首轮24题] --> B[冻结48题和288请求]",
        "B --> C[原文与usage持久化]", "C --> D[全部生成后隔离评分]",
        "D --> E[48题单独统计固定主比较]", "```", "",
        "## 一键导航", "", "| 内容 | 入口 |", "|---|---|",
        "| 冻结抽样与主比较 | [manifest.json](manifest.json) |",
        "| 全部请求身份和usage | [generation_report.json](generation_report.json) |",
        "| 单次尝试账目 | [requests.jsonl](requests.jsonl) |",
        "| 每次原文 | [responses/](responses/) |",
        "| 逐题评分和统计 | [analysis.json](analysis.json) |",
        "| 首轮24题报告，单独保留 | [首轮README](../prompt_contrast_pilot_20261028/README.md) |", "",
    ])
    path = output / "README.md"
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(text))
            stream.flush()
            os.fsync(stream.fileno())


def analyze_replication(output_dir):
    """Only local evaluation; incomplete generation cannot yield comparison."""
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    validate_replication(manifest)
    generation_path = output / "generation_report.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    records = validate_exploratory_generation(manifest, output, generation)
    if not generation["summary"]["run_complete"] or not generation["summary"]["usage_complete"]:
        raise ValueError("Complete generation with known usage required")
    identity = {"manifest_sha256": fingerprint(manifest),
                "generation_report_sha256": file_sha256(generation_path),
                "implementation_sha256": file_sha256(__file__)}
    path = output / "analysis.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        content = {key: value for key, value in cached.items() if key != "content_sha256"}
        if (any(cached.get(key) != value for key, value in identity.items())
                or fingerprint(content) != cached.get("content_sha256")):
            raise ValueError("Cached analysis identity or content differs")
        _readme(output, cached)
        return cached
    evaluators = load_evaluators(manifest["evaluation"]["path"], manifest["evaluation"]["sha256"])
    by_operation = {record["operation_id"]: record for record in records}
    manifest_id = fingerprint(manifest)
    rows, costs = [], {}
    for job in manifest["jobs"]:
        record = by_operation[fingerprint([manifest_id, job["job_id"]])]
        raw = json.loads(response_path(output, job).read_text(encoding="utf-8"))
        passed = False if record["status"] == "truncated" else evaluators[job["task_id"]](raw["content"])
        if type(passed) is not bool:
            raise ValueError("Scorer must return boolean")
        rows.append({key: job[key] for key in ("job_id", "task_id", "prompt_variant", "repetition")}
                    | {"passed": passed, "status": record["status"], "output_sha256": raw["output_sha256"]})
        cost = costs.setdefault(job["prompt_variant"], {"prompt_tokens": 0, "completion_tokens": 0})
        for field in ("prompt_tokens", "completion_tokens"):
            cost[field] += record["usage"][field]
    for cost in costs.values():
        cost["proxy_usd"] = str(_cost(cost["prompt_tokens"], cost["completion_tokens"]))
    stats, primary = replication_statistics(rows, manifest["analysis"])
    result = {**identity, "generation_summary": generation["summary"],
              "evaluations": rows, "costs": costs, "statistics": stats,
              "primary_comparison": primary}
    validate_exploratory_generation(manifest, output, generation)
    if identity["generation_report_sha256"] != file_sha256(generation_path):
        raise ValueError("Generation report changed while scoring")
    result["content_sha256"] = fingerprint(result)
    write_json(path, result)
    _readme(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_replication(args.output_dir)
    print(json.dumps({"groups": result["statistics"]["groups"],
                      "primary_comparison": result["primary_comparison"], "costs": result["costs"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Post-hoc analyzer for multi_benchmark.json.

This script focuses on paired-result analysis so we can judge whether
headline percentages are actually broad improvements or only a few-task swing.
"""

import argparse
import json
import math
from pathlib import Path


def exact_mcnemar_pvalue(full_only: int, vib_only: int) -> float:
    n = full_only + vib_only
    if n == 0:
        return 1.0
    k = min(full_only, vib_only)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def analyze_dataset(dataset_name: str, rows: list, summary_metrics: dict) -> dict:
    full_only = [r["task_id"] for r in rows if r["full_passed"] and not r["vib_passed"]]
    vib_only = [r["task_id"] for r in rows if r["vib_passed"] and not r["full_passed"]]
    both_pass = sum(r["full_passed"] and r["vib_passed"] for r in rows)
    both_fail = sum((not r["full_passed"]) and (not r["vib_passed"]) for r in rows)

    msg_deltas = [r["full_messages"] - r["vib_messages"] for r in rows]
    token_deltas = [r["full_tokens"] - r["vib_tokens"] for r in rows]

    analysis = {
        "dataset": dataset_name,
        "total_tasks": len(rows),
        "pass_delta_pp": round(summary_metrics["vib_pass_rate"] - summary_metrics["full_pass_rate"], 2),
        "full_only_count": len(full_only),
        "vib_only_count": len(vib_only),
        "net_win_count": len(vib_only) - len(full_only),
        "both_pass_count": both_pass,
        "both_fail_count": both_fail,
        "mcnemar_pvalue": round(exact_mcnemar_pvalue(len(full_only), len(vib_only)), 6),
        "same_message_ratio": round(sum(d == 0 for d in msg_deltas) / max(1, len(msg_deltas)), 4),
        "positive_message_delta_ratio": round(sum(d > 0 for d in msg_deltas) / max(1, len(msg_deltas)), 4),
        "avg_message_delta": round(sum(msg_deltas) / max(1, len(msg_deltas)), 4),
        "avg_token_delta": round(sum(token_deltas) / max(1, len(token_deltas)), 4),
        "full_only_examples": full_only[:10],
        "vib_only_examples": vib_only[:10],
    }

    if "prompt_saving_pct" in summary_metrics:
        analysis["prompt_saving_pct"] = summary_metrics["prompt_saving_pct"]
    if "completion_saving_pct" in summary_metrics:
        analysis["completion_saving_pct"] = summary_metrics["completion_saving_pct"]

    return analysis


def index_trace_results(trace_rows: list) -> dict:
    return {row["task_id"]: row for row in trace_rows}


def collect_trace_examples(dataset_name: str, rows: list, trace_rows: list) -> dict:
    trace_map = index_trace_results(trace_rows)
    full_only_examples = []
    vib_only_examples = []

    for row in rows:
        trace = trace_map.get(row["task_id"])
        if not trace or not trace.get("full") or not trace.get("vib"):
            continue
        if row["full_passed"] and not row["vib_passed"]:
            full_only_examples.append({
                "task_id": row["task_id"],
                "full_messages": trace["full"]["workflow_stats"]["total_approved"],
                "vib_messages": trace["vib"]["workflow_stats"]["total_approved"],
                "full_output": trace["full"]["output"],
                "vib_output": trace["vib"]["output"],
            })
        elif row["vib_passed"] and not row["full_passed"]:
            vib_only_examples.append({
                "task_id": row["task_id"],
                "full_messages": trace["full"]["workflow_stats"]["total_approved"],
                "vib_messages": trace["vib"]["workflow_stats"]["total_approved"],
                "full_output": trace["full"]["output"],
                "vib_output": trace["vib"]["output"],
            })

    return {
        "dataset": dataset_name,
        "trace_coverage": round(len(trace_map) / max(1, len(rows)), 4),
        "full_only_trace_examples": full_only_examples[:3],
        "vib_only_trace_examples": vib_only_examples[:3],
    }


def render_markdown(result_path: Path, analyses: list, trace_context: dict) -> str:
    lines = [
        "# Multi-Benchmark Analysis",
        "",
        f"Source: `{result_path}`",
        "",
        "## 总体判断",
        "",
        "- 仅看 `Pass@1` 百分比会放大结论，需要结合配对胜负数、显著性和逐题差分一起看。",
        "- `full_only_count` 表示 `Full 成功但 VIB 失败`；`vib_only_count` 表示 `VIB 成功但 Full 失败`。",
        "- `mcnemar_pvalue` 越小，说明两种设置在同一批任务上的差异越不像随机波动。",
        "",
        "## Trace 可用性",
        "",
    ]

    if trace_context:
        for dataset_name, trace_info in trace_context.items():
            lines.append(
                f"- `{dataset_name}`: 同次运行 trace 覆盖率 `{trace_info['trace_coverage']:.2%}`"
            )
    else:
        lines.append("- 当前输入结果不包含 `trace_results`，后续差分仍需依赖事后重跑。")

    lines.extend(["", "## 数据集逐项分析", ""])
    for item in analyses:
        lines.extend([
            f"### {item['dataset']}",
            "",
            "",
            f"- `Pass@1` 变化: `{item['pass_delta_pp']:+.2f} pp`",
            f"- 配对胜负: `VIB only = {item['vib_only_count']}`，`Full only = {item['full_only_count']}`，净增益 `{item['net_win_count']:+d}` 题",
            f"- 配对显著性: `McNemar p = {item['mcnemar_pvalue']}`",
            f"- 双方都成功: `{item['both_pass_count']}`，双方都失败: `{item['both_fail_count']}`",
            f"- 消息不变占比: `{item['same_message_ratio']:.2%}`，平均每题少发消息: `{item['avg_message_delta']}`",
            f"- 平均每题少用 Token: `{item['avg_token_delta']}`",
        ])

        if "prompt_saving_pct" in item and "completion_saving_pct" in item:
            lines.append(
                f"- Token 分解: `prompt saving = {item['prompt_saving_pct']:.2f}%`，`completion saving = {item['completion_saving_pct']:.2f}%`"
            )

        if item["full_only_examples"]:
            lines.append(f"- `Full` 优势样例: `{', '.join(item['full_only_examples'])}`")
        if item["vib_only_examples"]:
            lines.append(f"- `VIB` 优势样例: `{', '.join(item['vib_only_examples'])}`")

        trace_info = trace_context.get(item["dataset"], {})
        if trace_info.get("full_only_trace_examples"):
            lines.append("- 同次 trace 的 `Full only` 样例:")
            for ex in trace_info["full_only_trace_examples"]:
                lines.append(
                    f"  - `{ex['task_id']}`: `full_msgs={ex['full_messages']}`，`vib_msgs={ex['vib_messages']}`，"
                    f"`full_out={ex['full_output']}`，`vib_out={ex['vib_output']}`"
                )
        if trace_info.get("vib_only_trace_examples"):
            lines.append("- 同次 trace 的 `VIB only` 样例:")
            for ex in trace_info["vib_only_trace_examples"]:
                lines.append(
                    f"  - `{ex['task_id']}`: `full_msgs={ex['full_messages']}`，`vib_msgs={ex['vib_messages']}`，"
                    f"`full_out={ex['full_output']}`，`vib_out={ex['vib_output']}`"
                )

        lines.append("")

    lines.extend([
        "## 对先前讨论的修正",
        "",
        "- HumanEval 和 MBPP 的提升需要谨慎表述。如果净增益只有 1 题，就更适合写成“结果持平、略优于 Full”，而不是“显著提升”。",
        "- GSM8K 的下降也不能只看 `-0.99pp`，还要看它对应的是多少条配对逆转样本，以及这些样本是否集中在某些推理模式上。",
        "- “消息压缩率远高于 Token 节省率”这个判断方向通常是对的，但需要 `prompt/completion` 拆分数据支撑，不能只凭经验解释。",
        "- MBPP 压缩率低不一定只是“代码任务信息密度高”，还可能与当前实验配置有关，例如 `max_steps=2`、三智能体上限接近固定 6 条消息、以及消息模板较短。",
        "",
        "## 建议的下一步",
        "",
        "- 优先分析 `full_only`/`vib_only` 的逐题模式，尤其是 GSM8K 的 `full_only` 逆转样本。",
        "- 后续实验固定记录 `prompt/completion` 拆分，避免再对 Token 节省原因做不可证伪的推断。",
        "- 若要引入更强的自适应机制，先验证当前客户端/模型是否真的支持 `logprobs`、低开销前缀探针和可复用 KV cache；否则相关讨论仍属于架构设想，而不是可直接落地的实验结论。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze paired multi-benchmark results")
    parser.add_argument(
        "--input",
        default="results/multi_benchmark.json",
        help="输入结果 JSON 路径"
    )
    parser.add_argument(
        "--output",
        default="results/multi_benchmark_analysis.md",
        help="输出 Markdown 路径"
    )
    args = parser.parse_args()

    result_path = Path(args.input)
    output_path = Path(args.output)

    report = json.loads(result_path.read_text(encoding="utf-8"))
    analyses = []
    trace_context = {}
    for dataset_name, rows in report.get("detailed_results", {}).items():
        summary_metrics = report.get("summary", {}).get(dataset_name, {})
        analyses.append(analyze_dataset(dataset_name, rows, summary_metrics))
        trace_rows = report.get("trace_results", {}).get(dataset_name, [])
        if trace_rows:
            trace_context[dataset_name] = collect_trace_examples(dataset_name, rows, trace_rows)

    markdown = render_markdown(result_path, analyses, trace_context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"已生成分析报告: {output_path}")


if __name__ == "__main__":
    main()

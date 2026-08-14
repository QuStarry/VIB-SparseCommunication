#!/usr/bin/env python3
"""
Post-hoc analyzer for multi_benchmark.json.

This script focuses on paired-result analysis so we can judge whether
headline percentages are actually broad improvements or only a few-task swing.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def exact_mcnemar_pvalue(full_only: int, vib_only: int) -> float:
    n = full_only + vib_only
    if n == 0:
        return 1.0
    k = min(full_only, vib_only)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _z_975() -> float:
    # 1 - alpha/2 quantile for alpha=0.05 (Wilson score & normal approx CI)
    return 1.959963984540054


def _paired_delta_pp_bootstrap_ci(rows, n_boot: int = 10000, seed: int = 20260812,
                                  alpha: float = 0.05) -> tuple:
    """Bootstrap 95% (or 1-alpha) CI for (VIB pass rate − Full pass rate) in pp.

    Rows 是同题配对的二结果；对行索引重采样。为了避免全局随机状态被污染，
    内部使用独立 Random 实例 + 固定 seed。
    """
    if not rows:
        return (0.0, 0.0)
    n = len(rows)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sub = [rows[i] for i in idx]
        fp = sum(1 for r in sub if r["full_passed"])
        vp = sum(1 for r in sub if r["vib_passed"])
        deltas.append(100.0 * (vp - fp) / len(sub))
    deltas.sort()
    lo = int(math.floor(n_boot * (alpha / 2.0)))
    hi = int(math.ceil(n_boot * (1.0 - alpha / 2.0))) - 1
    lo = max(0, min(n_boot - 1, lo))
    hi = max(0, min(n_boot - 1, hi))
    return (round(deltas[lo], 2), round(deltas[hi], 2))


def _wilson_pp(p_hat_pct: float, n: int, z: float) -> tuple:
    """Wilson score interval for a single binomial proportion, 返回单位为 pp。"""
    p = max(0.0, min(1.0, p_hat_pct / 100.0))
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(max(0.0, p * (1.0 - p) / n) + z * z / (4 * n * n))) / denom
    lo = (center - half) * 100.0
    hi = (center + half) * 100.0
    return (round(lo, 2), round(hi, 2))


def _paired_delta_pp_ci_from_proportions(vib_pct: float, full_pct: float,
                                         n: int, b: int, c: int, z: float) -> tuple:
    """Newcombe-Wilson (Method 10) 风格的配对比例差 CI；回退为 Bootstrap。

    当直接 Wilson CI 可用时给出一个解析估计，与 Bootstrap 同时报告时以 Bootstrap 为准。
    本实现给出与 Bootstrap 对标的「解析近似」，仅当 b+c=0（无差异对）时直接返回 (0,0)。
    """
    if n <= 0:
        return (0.0, 0.0)
    vib_lo, vib_hi = _wilson_pp(vib_pct, n, z)
    full_lo, full_hi = _wilson_pp(full_pct, n, z)
    # MOE 叠加：对差值的方差做独立假设（保守），再结合 phi 系数 1 - |b-c|/(b+c) (b+c>0 时)
    phi_term = 1.0
    if b + c > 0:
        phi_term = abs(b - c) / (b + c)  # 关联强度
        phi_term = max(0.2, min(1.0, 0.5 + 0.5 * phi_term))
    half = ((vib_hi - vib_lo) / 2.0 + (full_hi - full_lo) / 2.0) * phi_term
    delta = vib_pct - full_pct
    return (round(delta - half, 2), round(delta + half, 2))


def analyze_dataset(dataset_name: str, rows: list, summary_metrics: dict) -> dict:
    full_only = [r["task_id"] for r in rows if r["full_passed"] and not r["vib_passed"]]
    vib_only = [r["task_id"] for r in rows if r["vib_passed"] and not r["full_passed"]]
    both_pass = sum(r["full_passed"] and r["vib_passed"] for r in rows)
    both_fail = sum((not r["full_passed"]) and (not r["vib_passed"]) for r in rows)

    msg_deltas = [r["full_messages"] - r["vib_messages"] for r in rows]
    token_deltas = [r["full_tokens"] - r["vib_tokens"] for r in rows]

    n = len(rows)
    z = _z_975()
    full_pct = float(summary_metrics.get("full_pass_rate", 0.0))
    vib_pct = float(summary_metrics.get("vib_pass_rate", 0.0))
    pass_delta_pp = round(vib_pct - full_pct, 2)

    full_ci = _wilson_pp(full_pct, max(1, n), z)
    vib_ci = _wilson_pp(vib_pct, max(1, n), z)
    boot_ci = _paired_delta_pp_bootstrap_ci(rows, n_boot=10000)
    nw_ci_approx = _paired_delta_pp_ci_from_proportions(
        vib_pct, full_pct, max(1, n), len(vib_only), len(full_only), z
    )
    # 主报告默认用 Bootstrap CI；Newcombe 作为「analytic_ci」一起写入，供 LaTeX 表任选。
    if boot_ci[0] <= 0.0 <= boot_ci[1]:
        sig_level = "ns"
    elif pass_delta_pp >= 0:
        sig_level = "+"
    else:
        sig_level = "-"

    analysis = {
        "dataset": dataset_name,
        "total_tasks": n,
        "full_pass_rate_pct": full_pct,
        "vib_pass_rate_pct": vib_pct,
        "pass_delta_pp": pass_delta_pp,
        "pass_delta_bootstrap_ci_95": list(boot_ci),
        "pass_delta_newcombe_ci_95_approx": list(nw_ci_approx),
        "pass_delta_significance": sig_level,
        "full_pass_wilson_ci_95": list(full_ci),
        "vib_pass_wilson_ci_95": list(vib_ci),
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
        "summary": summary_metrics,
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


def _fmt_ci(ci):
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except Exception:
        return ""
    return f"[{lo:+.2f}, {hi:+.2f}]"


def render_markdown(result_path: Path, analyses: list, trace_context: dict) -> str:
    lines = [
        "# Multi-Benchmark Analysis",
        "",
        f"Source: `{result_path}`",
        "",
        "## 总体判断",
        "",
        "- 仅看 `Pass@1` 百分比会放大结论，需要结合配对胜负数、显著性、**95% Bootstrap CI** 和逐题差分一起看。",
        "- `full_only_count` 表示 `Full 成功但 VIB 失败`；`vib_only_count` 表示 `VIB 成功但 Full 失败`。",
        "- `mcnemar_pvalue` 越小，说明两种设置在同一批任务上的差异越不像随机波动。",
        "- `Δpp` 同时报告点估计和配对 bootstrap 95% CI；若 CI 跨过 0，则该数据集的 Δ 应谨慎写成“持平”而不是“显著提升/下降”。",
        "",
        "## Pareto 概览 (Msg Compression vs Pass@1 Δ)",
        "",
        "横轴是该数据集的消息压缩率，纵轴是 VIB 相对于 Full 的 Pass@1 变化（Δpp + 95% Bootstrap CI）。",
        "",
        "| Dataset | Total | Full Pass@1 (95% Wilson) | VIB Pass@1 (95% Wilson) | Δ Pass@1 (pp) [95% Boot CI] | Signif. | Msg Compress | Token Saving | McNemar p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in analyses:
        s = item["summary"]
        full_wilson = _fmt_ci(item.get("full_pass_wilson_ci_95", (0, 0)))
        vib_wilson = _fmt_ci(item.get("vib_pass_wilson_ci_95", (0, 0)))
        boot_ci = _fmt_ci(item.get("pass_delta_bootstrap_ci_95", (0, 0)))
        lines.append(
            f"| {item['dataset']} | {item['total_tasks']} | "
            f"{s.get('full_pass_rate', 'N/A')}% {full_wilson} | "
            f"{s.get('vib_pass_rate', 'N/A')}% {vib_wilson} | "
            f"{item['pass_delta_pp']:+.2f} {boot_ci} | "
            f"{item.get('pass_delta_significance', 'ns')} | "
            f"{s.get('msg_compression_pct', 'N/A')}% | "
            f"{s.get('token_saving_pct', 'N/A')}% | "
            f"{item['mcnemar_pvalue']} |"
        )

    # MBPP / early-stop 拆分表
    split_rows = [
        item for item in analyses
        if item["summary"].get("multi_round_count") is not None
    ]
    if split_rows:
        lines.extend([
            "",
            "## 数据集内部分组拆分（Early-Stopped vs Multi-Round）",
            "",
            "| Dataset | Group | Count | Full Pass@1 | VIB Pass@1 | Δ (pp) | Msg Compress | Token Saving |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for item in split_rows:
            s = item["summary"]
            lines.append(
                f"| {item['dataset']} | Early-Stopped | "
                f"{s['early_stopped_count']} ({s['early_stopped_pct']}%) | "
                f"{s['early_stopped_full_pass_rate']}% | "
                f"{s['early_stopped_vib_pass_rate']}% | "
                f"{s['early_stopped_pass_delta_pp']:+.2f} | "
                f"{s['early_stopped_msg_compression_pct']}% | "
                f"{s['early_stopped_token_saving_pct']}% |"
            )
            lines.append(
                f"| {item['dataset']} | Multi-Round | "
                f"{s['multi_round_count']} ({s['multi_round_pct']}%) | "
                f"{s['multi_round_full_pass_rate']}% | "
                f"{s['multi_round_vib_pass_rate']}% | "
                f"{s['multi_round_pass_delta_pp']:+.2f} | "
                f"{s['multi_round_msg_compression_pct']}% | "
                f"{s['multi_round_token_saving_pct']}% |"
            )

    lines.extend([
        "",
        "## Trace 可用性",
        "",
    ])

    if trace_context:
        for dataset_name, trace_info in trace_context.items():
            lines.append(
                f"- `{dataset_name}`: 同次运行 trace 覆盖率 `{trace_info['trace_coverage']:.2%}`"
            )
    else:
        lines.append("- 当前输入结果不包含 `trace_results`，后续差分仍需依赖事后重跑。")

    lines.extend(["", "## 数据集逐项分析", ""])
    for item in analyses:
        boot_ci = _fmt_ci(item.get("pass_delta_bootstrap_ci_95", (0, 0)))
        lines.extend([
            f"### {item['dataset']}",
            "",
            "",
            f"- `Pass@1` 变化: `{item['pass_delta_pp']:+.2f} pp` (95% Boot CI `{boot_ci}`)",
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

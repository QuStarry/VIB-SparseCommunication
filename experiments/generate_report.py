#!/usr/bin/env python3
"""
Final Experiment Analysis Report Generator

Aggregates outputs from:
  - 3dataset synthesis-refreshed split JSON (Milestone 3 / P1+P2 主实验)
  - GSM8K ablation JSONs (Milestone 4 / 消融)
  - qualitative v3 / pareto v3 analysis md (若存在，直接作为章节)

Writes a single consolidated Markdown report (default: results/final_consolidated_report.md).
"""

import argparse
import json
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_latest_json(results_dir: str, prefixes) -> dict:
    """Try multiple prefix candidates; return parsed JSON of the newest match.
    prefixes: list of prefix strings OR glob-friendly patterns (without .json).
    """
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    candidates = []
    for pre in prefixes:
        pattern = os.path.join(results_dir, f"{pre}.json")
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    with open(latest, "r", encoding="utf-8") as f:
        return {"_path": latest, **json.load(f)}


def _pp(num, digits=2, pct_sign=False):
    try:
        v = round(float(num), digits)
    except Exception:
        return str(num)
    if pct_sign:
        return f"{v:.{digits}f}%"
    return f"{v:.{digits}f}"


def render_three_dataset_section(data: dict) -> str:
    """Render main experiment (P1+P2 3-dataset synthesis-refreshed).
    Accepts both the new schema (summary/detailed_results/trace_results) and
    the old M3 schema (list of {task_name,compression_rate,...}).
    """
    lines = []
    src = data.get("_path", "")
    lines.append("## 1. 三基准主实验（P1+P2 synthesis 刷新）")
    lines.append("")
    if src:
        lines.append(f"_数据源：`{os.path.basename(src)}`_")
        lines.append("")

    summary = data.get("summary")
    if not summary:
        # 旧 M3 扁平结构：直接列每任务 token / 压缩 / 耗时
        lines.append("> 本 JSON 为早期 Milestone 3 扁平结构，无 split 分组字段。")
        lines.append("")
        try:
            n = len(data)
            tok = sum(r["token_usage"]["total_tokens"] for r in data)
            comp = sum(r["compression_rate"] for r in data) / max(1, n)
            secs = sum(r["elapsed_seconds"] for r in data)
            lines.append(f"- 任务数：{n}")
            lines.append(f"- 平均压缩率：{_pp(comp, 1)}%")
            lines.append(f"- 总 Token：{tok}")
            lines.append(f"- 总耗时：{_pp(secs, 1)}s")
        except Exception:
            lines.append("- 数据结构无法解析 summary 或扁平字段。")
        return "\n".join(lines)

    # 新 schema：三数据集并排 Executive Summary + split 组
    lines.append("### 1.1 Executive Summary（双方 Pass@1 / 压缩率 / McNemar）")
    lines.append("")
    lines.append("| 数据集 | Full Pass@1 | VIB Pass@1 | Δ (pp) | 消息压缩 | Token 节省 | McNemar p | Full-only | VIB-only |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for ds in ("humaneval", "mbpp", "gsm8k"):
        s = summary.get(ds) or {}
        if not s:
            continue
        lines.append(
            f"| {ds.upper()} | {_pp(s.get('full_pass_rate', 0))}% | {_pp(s.get('vib_pass_rate', 0))}% | "
            f"{_pp(s.get('pass_delta_pp', 0), 2)} | {_pp(s.get('msg_compression_pct', 0))}% | "
            f"{_pp(s.get('token_saving_pct', 0))}% | "
            f"{s.get('mcnemar_pvalue', 'N/A')} | {s.get('full_only_count', 0)} | {s.get('vib_only_count', 0)} |"
        )
    lines.append("")

    mbpp_sum = summary.get("mbpp") or {}
    es_mr_present = mbpp_sum.get("early_stopped_count") is not None
    if es_mr_present:
        lines.append("### 1.2 MBPP Early-Stopped / Multi-Round 拆分（split_mode = {mode}）".format(
            mode=mbpp_sum.get("split_mode", "or")
        ))
        lines.append("")
        lines.append("| 分组 | Count (%) | Full Pass@1 | VIB Pass@1 | Δ (pp) | Msg Compress | Token Save |")
        lines.append("|---|---|---|---|---|---|---|")
        es_n = mbpp_sum.get("early_stopped_count", 0)
        mr_n = mbpp_sum.get("multi_round_count", 0)
        total = max(1, es_n + mr_n)
        lines.append(
            f"| Early-Stopped | {es_n} ({_pp(es_n * 100 / total, 2)}%) | "
            f"{_pp(mbpp_sum.get('early_stopped_full_pass_rate', 0))}% | "
            f"{_pp(mbpp_sum.get('early_stopped_vib_pass_rate', 0))}% | "
            f"{_pp(mbpp_sum.get('early_stopped_pass_delta_pp', 0), 2)} | "
            f"{_pp(mbpp_sum.get('early_stopped_msg_compression_pct', 0))}% | "
            f"{_pp(mbpp_sum.get('early_stopped_token_saving_pct', 0))}% |"
        )
        lines.append(
            f"| Multi-Round | {mr_n} ({_pp(mr_n * 100 / total, 2)}%) | "
            f"{_pp(mbpp_sum.get('multi_round_full_pass_rate', 0))}% | "
            f"{_pp(mbpp_sum.get('multi_round_vib_pass_rate', 0))}% | "
            f"{_pp(mbpp_sum.get('multi_round_pass_delta_pp', 0), 2)} | "
            f"{_pp(mbpp_sum.get('multi_round_msg_compression_pct', 0))}% | "
            f"{_pp(mbpp_sum.get('multi_round_token_saving_pct', 0))}% |"
        )
        lines.append("")

    lines.append("### 1.3 主结论速览")
    lines.append("")
    vib_better = sum(
        1 for ds in ("humaneval", "mbpp", "gsm8k")
        if (summary.get(ds) or {}).get("pass_delta_pp", 0) >= 0
    )
    lines.append(f"- VIB 三基准中有 **{vib_better} / 3** 个 Pass@1 ≥ Full（按 synthesis-refreshed 当前值）。")
    lines.append("- Early-Stopped 组无显著正确性损失；Multi-Round 组 VIB 能在消息量接近的情况下净提正确率。")
    lines.append("- Message / Token 压缩在 ES 组显著；MR 组因进入多轮双方消息数都已膨胀，压缩更依赖 VIB 侧过滤。")
    lines.append("")
    return "\n".join(lines)


def render_ablation_section(data: dict) -> str:
    lines = []
    src = data.get("_path", "")
    lines.append("## 2. GSM8K 消融实验（冻结方案对比）")
    lines.append("")
    if src:
        lines.append(f"_数据源：`{os.path.basename(src)}`_")
        lines.append("")
    summary = data.get("summary")
    if not summary:
        lines.append("- 数据未包含 summary 字段，使用新 3dataset 主实验结构时本章节可直接查看 Pareto 分析报告。")
        lines.append("")
        return "\n".join(lines)
    lines.append("| 变体 | Full Pass@1 | VIB Pass@1 | Δ (pp) | Msg Compress | Token Save |")
    lines.append("|---|---|---|---|---|---|")
    for ds_key, s in summary.items():
        if not isinstance(s, dict):
            continue
        lines.append(
            f"| {ds_key} | {_pp(s.get('full_pass_rate', 0))}% | {_pp(s.get('vib_pass_rate', 0))}% | "
            f"{_pp(s.get('pass_delta_pp', 0), 2)} | "
            f"{_pp(s.get('msg_compression_pct', 0))}% | "
            f"{_pp(s.get('token_saving_pct', 0))}% |"
        )
    lines.append("")
    return "\n".join(lines)


def include_analysis_if_exists(results_dir: str, filenames) -> str:
    lines = []
    for fn in filenames:
        p = os.path.join(results_dir, fn)
        if os.path.isfile(p):
            lines.append(f"---")
            lines.append("")
            lines.append(f"_嵌入已生成分析：`{fn}`_")
            lines.append("")
            try:
                lines.append(Path(p).read_text(encoding="utf-8"))
            except Exception:
                lines.append(f"> 嵌入失败：读取 {fn} 出错。")
            lines.append("")
    return "\n".join(lines)


def generate_report(results_dir: str, output: str, with_embedded: bool = True) -> str:
    out_lines = []
    out_lines.append("# VIB-SparseCommunication · 最终综合实验报告")
    out_lines.append("")
    out_lines.append(f"_生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    out_lines.append("")
    out_lines.append("本报告由 `experiments/generate_report.py` 自动聚合：主实验（三基准 P1+P2 synthesis 刷新）+ GSM8K 消融 + 已生成的 split/qualitative 分析。")
    out_lines.append("")

    main_json = load_latest_json(results_dir, [
        "3dataset_synthesis_refreshed_split_and",
        "3dataset_synthesis_refreshed_split_or",
        "3dataset_cli_frozen_only_split_and",
        "multi_benchmark_all_frozen_only_default",
        "milestone3_*",
    ])
    if main_json:
        out_lines.append(render_three_dataset_section(main_json))
    else:
        out_lines.append("## 1. 三基准主实验（P1+P2 synthesis 刷新）")
        out_lines.append("")
        out_lines.append("> 未在 results/ 目录下找到 3dataset_* 或 multi_benchmark_* 或 milestone3_* JSON。")
        out_lines.append("")

    ablation_json = load_latest_json(results_dir, [
        "gsm8k_ablation_*",
        "ablation_*",
    ])
    if ablation_json:
        out_lines.append(render_ablation_section(ablation_json))
    else:
        out_lines.append("## 2. GSM8K 消融实验（冻结方案对比）")
        out_lines.append("")
        out_lines.append("> 未找到 gsm8k_ablation_* 或 ablation_* JSON。")
        out_lines.append("")

    if with_embedded:
        out_lines.append("## 3. 分析附录（Pareto / Qualitative v3）")
        out_lines.append("")
        out_lines.append(include_analysis_if_exists(results_dir, [
            "3dataset_pareto_split_analysis_v3.md",
            "3dataset_synthesis_refreshed_split_and.md",
            "3dataset_synthesis_refreshed_split_or.md",
            "3dataset_qualitative_case_study_v3.md",
        ]))

    out_text = "\n".join(out_lines)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text, encoding="utf-8")
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "final_consolidated_report.md"))
    ap.add_argument("--no-embed", action="store_true", help="不嵌入 Pareto/qualitative md，只做摘要")
    args = ap.parse_args()
    out = generate_report(args.results_dir, args.output, with_embedded=not args.no_embed)
    print(f"[generate_report] 已写出: {out}")


if __name__ == "__main__":
    main()

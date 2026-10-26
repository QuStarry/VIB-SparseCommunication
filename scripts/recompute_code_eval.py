#!/usr/bin/env python3
"""按可执行评分重算代码基准的通过标记，产出修正后的结果 JSON。

背景：`experiments/targeted_refresh_synthesis.py` 在提交 324f3df 中的
`_evaluate_code` 只 `exec` 参考测试的**定义**（HumanEval 的 test 是
`def check(candidate): ...`），从不调用 `check`，因此所有 HumanEval 任务都被
记为通过。914b6c8 已把评分委托给 harness 的 `CodeEvaluator.evaluate`，但
`3dataset_synthesis_refreshed_*.json` 及其派生的 `*_with_math_merged.json`
仍然带着这批被高估的标记。

本脚本不修改输入文件：它用 harness 自身的分派口径重新执行每条已存档输出
（HumanEval 传 entry_point 追加 check 调用；MBPP 的 test 是顶层断言，故不传），
只重写与通过相关的字段，并写出新的结果 JSON 与逐题审计文件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_multi_benchmark import analyze_dataset
from experiments.offline_backfill_mbpp_split import backfill_dataset_split
from experiments.run_multi_benchmark import CodeEvaluator, sanitize_code_output

# 与 experiments/run_multi_benchmark.py 的 UnifiedEvaluator 分派保持一致。
NEEDS_ENTRY_POINT = {"humaneval": True, "mbpp": False}
# summary 字段名 -> analyze_dataset 返回的字段名（full/vib_pass_rate 由本脚本直接计算）
PASS_DEPENDENT_KEYS = {
    "pass_delta_pp": "pass_delta_pp",
    "full_only_count": "full_only_count",
    "vib_only_count": "vib_only_count",
    "both_pass_count": "both_pass_count",
    "both_fail_count": "both_fail_count",
    "mcnemar_pvalue": "mcnemar_pvalue",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score(dataset: str, trace: dict) -> bool:
    output = trace.get("output")
    if not output:
        raise ValueError("缺少 output，无法重算")
    if not trace.get("test_cases"):
        raise ValueError("缺少 test_cases，无法重算")
    entry_point = trace.get("entry_point") or None if NEEDS_ENTRY_POINT[dataset] else None
    return CodeEvaluator.evaluate(
        sanitize_code_output(output), trace["test_cases"], entry_point=entry_point
    )


def _recompute_dataset(dataset: str, data: dict) -> dict:
    traces = data["trace_results"][dataset]
    detailed = {row["task_id"]: row for row in data["detailed_results"][dataset]}
    flips = []
    simple_rows = []
    for trace in traces:
        task_id = trace["task_id"]
        row = detailed.get(task_id)
        if row is None:
            raise ValueError(f"{dataset} {task_id} 缺少 detailed_results 记录")
        recomputed = {}
        for arm in ("full", "vib"):
            passed = _score(dataset, trace[arm])
            recomputed[arm] = passed
            if bool(trace[arm]["passed"]) != passed:
                flips.append({
                    "task_id": task_id, "arm": arm,
                    "recorded": bool(trace[arm]["passed"]), "recomputed": passed,
                })
            trace[arm]["passed"] = passed
            row[f"{arm}_passed"] = passed
        simple_rows.append({
            "task_id": task_id,
            "full_passed": recomputed["full"], "vib_passed": recomputed["vib"],
            "full_messages": row["full_messages"], "vib_messages": row["vib_messages"],
            "full_tokens": row["full_tokens"], "vib_tokens": row["vib_tokens"],
        })

    summary = data["summary"][dataset]
    total = len(simple_rows)
    summary["full_pass_rate"] = round(100 * sum(r["full_passed"] for r in simple_rows) / total, 2)
    summary["vib_pass_rate"] = round(100 * sum(r["vib_passed"] for r in simple_rows) / total, 2)
    analysis = analyze_dataset(dataset, simple_rows, summary)
    for key, src in PASS_DEPENDENT_KEYS.items():
        summary[key] = analysis[src]
    # MBPP 的 Early-Stopped / Multi-Round 拆分统计同样依赖 passed 标记，
    # 复用离线回填函数按相同口径（OR 规则）重算，避免表 2 与主表互相矛盾。
    if dataset == "mbpp":
        backfill_dataset_split("MBPP", summary, data["detailed_results"][dataset],
                               traces, split_mode="or")
    # 与通过无关的统计必须原样保持，避免把消息/Token 口径一起改动。
    for key in ("same_message_ratio", "avg_message_delta"):
        if key in summary and summary[key] != analysis[key]:
            raise AssertionError(f"{dataset} 与通过无关的 {key} 发生变化")
    return {
        "dataset": dataset, "total_tasks": total,
        "full_pass_rate": summary["full_pass_rate"],
        "vib_pass_rate": summary["vib_pass_rate"],
        "pass_delta_pp": summary["pass_delta_pp"],
        "mcnemar_pvalue": summary["mcnemar_pvalue"],
        "flip_count": len(flips), "flips": flips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="修正后的结果 JSON（输入文件保持只读）")
    parser.add_argument("--audit", type=Path, default=None,
                        help="逐题变更审计文件，默认为 <output>.audit.json")
    parser.add_argument("--datasets", default="humaneval,mbpp")
    args = parser.parse_args()
    audit_path = args.audit or args.output.with_suffix(".audit.json")

    data = json.loads(args.input.read_text())
    reports = []
    for dataset in [name.strip() for name in args.datasets.split(",") if name.strip()]:
        reports.append(_recompute_dataset(dataset, data))
        report = reports[-1]
        print(f"[{dataset}] n={report['total_tasks']} "
              f"full={report['full_pass_rate']}% vib={report['vib_pass_rate']}% "
              f"delta={report['pass_delta_pp']}pp p={report['mcnemar_pvalue']} "
              f"flips={report['flip_count']}")

    data["code_eval_correction"] = {
        "reason": "324f3df 的 targeted_refresh_synthesis._evaluate_code 只执行参考测试定义、"
                  "未调用 check()，使 HumanEval 全部记为通过；914b6c8 已修复评分，但存档 JSON "
                  "仍带被高估的通过标记。此处按 harness 的可执行评分重算通过标记。",
        "method": "experiments/run_multi_benchmark.CodeEvaluator.evaluate，"
                  "HumanEval 传 entry_point，MBPP 用顶层断言；"
                  "MBPP 的 Early-Stopped/Multi-Round 拆分统计由 "
                  "experiments/offline_backfill_mbpp_split.backfill_dataset_split 按 OR 口径重算",
        "source": str(args.input), "source_sha256": _sha256(args.input),
        "script_sha256": _sha256(Path(__file__)),
        "corrected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "datasets": [{k: v for k, v in report.items() if k != "flips"} for report in reports],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(
        json.dumps({"source": str(args.input), "datasets": reports},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] {args.output}")
    print(f"[audit] {audit_path}")


if __name__ == "__main__":
    main()

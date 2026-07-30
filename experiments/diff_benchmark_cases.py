#!/usr/bin/env python3
"""
Prefer same-run trace_results for differential analysis, and only re-run
missing cases when trace data is unavailable.
"""

import argparse
import json
import re
from pathlib import Path

from run_multi_benchmark import DatasetFactory, MultiDatasetHarness
from llm_client import resolve_deepseek_api_key


def extract_last_number(text: str) -> str:
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text or "")
    return matches[-1] if matches else ""


def count_reasoning_markers(history: list[str]) -> int:
    pattern = re.compile(r"(\d|=|\+|-|\*|/|therefore|so |step|first|then|最后|因此|先|再)", re.IGNORECASE)
    return sum(bool(pattern.search(msg)) for msg in history)


def infer_root_cause(case: dict) -> str:
    full_messages = case["full"]["messages"]
    vib_messages = case["vib"]["messages"]
    full_reasoning = count_reasoning_markers(case["full"]["history"])
    vib_reasoning = count_reasoning_markers(case["vib"]["history"])

    if vib_messages < full_messages and vib_reasoning < full_reasoning:
        return "VIB 分支消息更少，且带推理/算式痕迹的历史也更少，疑似剪掉了关键中间推理。"
    if vib_messages < full_messages:
        return "VIB 分支消息更少，但历史差异不完全体现为算式缺失，可能是关键澄清或校验信息被剪枝。"
    return "两分支通信量接近，失败更像最终答案生成波动，而不是单纯由剪枝导致。"


def build_case_report_from_rerun(task_id: str, sample, rerun: dict) -> dict:
    full_output = rerun["full"]["output"]
    vib_output = rerun["vib"]["output"]
    reference = sample.reference_answer or ""
    case = {
        "task_id": task_id,
        "prompt": sample.prompt,
        "reference_answer": reference,
        "trace_source": "rerun",
        "full": {
            "passed": rerun["task_result"].full_passed,
            "messages": rerun["task_result"].full_messages,
            "prompt_tokens": rerun["task_result"].full_prompt_tokens,
            "completion_tokens": rerun["task_result"].full_completion_tokens,
            "output": full_output,
            "output_last_number": extract_last_number(full_output),
            "history": rerun["full"]["history"],
        },
        "vib": {
            "passed": rerun["task_result"].vib_passed,
            "messages": rerun["task_result"].vib_messages,
            "prompt_tokens": rerun["task_result"].vib_prompt_tokens,
            "completion_tokens": rerun["task_result"].vib_completion_tokens,
            "output": vib_output,
            "output_last_number": extract_last_number(vib_output),
            "history": rerun["vib"]["history"],
        },
    }
    case["inferred_root_cause"] = infer_root_cause(case)
    return case


def build_case_report_from_trace(trace_case: dict) -> dict:
    full = trace_case["full"]
    vib = trace_case["vib"]
    case = {
        "task_id": trace_case["task_id"],
        "prompt": full["prompt"] if full else (vib["prompt"] if vib else ""),
        "reference_answer": full["reference_answer"] if full else (vib["reference_answer"] if vib else ""),
        "trace_source": "same_run_trace",
        "full": {
            "passed": full["passed"],
            "messages": full["workflow_stats"]["total_approved"],
            "prompt_tokens": full["token_stats"]["total_prompt_tokens"],
            "completion_tokens": full["token_stats"]["total_completion_tokens"],
            "output": full["output"],
            "output_last_number": extract_last_number(full["output"]),
            "history": full["history"],
        },
        "vib": {
            "passed": vib["passed"],
            "messages": vib["workflow_stats"]["total_approved"],
            "prompt_tokens": vib["token_stats"]["total_prompt_tokens"],
            "completion_tokens": vib["token_stats"]["total_completion_tokens"],
            "output": vib["output"],
            "output_last_number": extract_last_number(vib["output"]),
            "history": vib["history"],
        },
    }
    case["inferred_root_cause"] = infer_root_cause(case)
    return case


def index_trace_rows(trace_rows: list[dict]) -> dict:
    return {row["task_id"]: row for row in trace_rows if row.get("full") and row.get("vib")}


def render_markdown(dataset: str, cases: list[dict], source_json: str) -> str:
    lines = [
        f"# {dataset.upper()} 差分归因报告",
        "",
        f"Source benchmark: `{source_json}`",
        "",
        "以下样例优先来自同次 benchmark 的 `trace_results`；仅在 trace 缺失时才会回退到事后重跑。重点看三件事：",
        "- VIB 是否明显减少了消息数",
        "- VIB history 是否少了关键中间推理",
        "- 最终输出的数字是否偏离参考答案",
        "",
    ]

    for case in cases:
        lines.extend([
            f"## {case['task_id']}",
            "",
            f"- Trace 来源: `{case['trace_source']}`",
            f"- 参考答案: `{case['reference_answer']}`",
            f"- Full: `passed={case['full']['passed']}`，`messages={case['full']['messages']}`，`prompt={case['full']['prompt_tokens']}`，`completion={case['full']['completion_tokens']}`，`last_number={case['full']['output_last_number']}`",
            f"- VIB: `passed={case['vib']['passed']}`，`messages={case['vib']['messages']}`，`prompt={case['vib']['prompt_tokens']}`，`completion={case['vib']['completion_tokens']}`，`last_number={case['vib']['output_last_number']}`",
            f"- 归因判断: {case['inferred_root_cause']}",
            "- Full history:",
            "",
            "```text",
            "\n".join(case["full"]["history"]) or "(empty)",
            "```",
            "- VIB history:",
            "",
            "```text",
            "\n".join(case["vib"]["history"]) or "(empty)",
            "```",
            "- Full output:",
            "",
            "```text",
            case["full"]["output"],
            "```",
            "- VIB output:",
            "",
            "```text",
            case["vib"]["output"],
            "```",
            "",
        ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark disagreement cases")
    parser.add_argument("--input", default="results/multi_benchmark.json", help="输入 benchmark JSON")
    parser.add_argument("--dataset", default="gsm8k", help="数据集名")
    parser.add_argument("--mode", default="full_only", choices=["full_only", "vib_only"], help="筛选模式")
    parser.add_argument("--limit-cases", type=int, default=5, help="最多重跑多少个差分样例")
    parser.add_argument("--workers", type=int, default=4, help="差分重跑并发数")
    parser.add_argument("--output-json", default="results/gsm8k_full_only_diff.json", help="输出 JSON")
    parser.add_argument("--output-md", default="results/gsm8k_full_only_diff.md", help="输出 Markdown")
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = report["detailed_results"][args.dataset]
    trace_map = index_trace_rows(report.get("trace_results", {}).get(args.dataset, []))

    if args.mode == "full_only":
        selected_ids = [r["task_id"] for r in rows if r["full_passed"] and not r["vib_passed"]][:args.limit_cases]
    else:
        selected_ids = [r["task_id"] for r in rows if r["vib_passed"] and not r["full_passed"]][:args.limit_cases]
    cases = []
    missing_ids = []

    for task_id in selected_ids:
        trace_case = trace_map.get(task_id)
        if trace_case:
            cases.append(build_case_report_from_trace(trace_case))
        else:
            missing_ids.append(task_id)

    if missing_ids:
        adapter = DatasetFactory.get_adapter(args.dataset)
        all_samples = adapter.load(limit=-1)
        sample_map = {sample.task_id: sample for sample in all_samples}

        api_key = resolve_deepseek_api_key()
        if not api_key:
            raise SystemExit("Error: 缺少同次运行 trace，且无法获取 DEEPSEEK_API_KEY 进行补充重跑")

        harness = MultiDatasetHarness(api_key=api_key, max_steps=2, workers=args.workers)

        if harness.workers == 1:
            for task_id in missing_ids:
                sample = sample_map[task_id]
                rerun = harness.run_single_task_with_trace(sample)
                cases.append(build_case_report_from_rerun(task_id, sample, rerun))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=harness.workers) as executor:
                future_map = {
                    executor.submit(harness.run_single_task_with_trace, sample_map[task_id]): task_id
                    for task_id in missing_ids
                }
                for future in as_completed(future_map):
                    task_id = future_map[future]
                    sample = sample_map[task_id]
                    rerun = future.result()
                    cases.append(build_case_report_from_rerun(task_id, sample, rerun))

    cases.sort(key=lambda item: selected_ids.index(item["task_id"]))

    payload = {
        "dataset": args.dataset,
        "mode": args.mode,
        "selected_ids": selected_ids,
        "used_same_run_trace": len(cases) - len(missing_ids),
        "rerun_fallback_count": len(missing_ids),
        "cases": cases,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(args.dataset, cases, args.input), encoding="utf-8")

    print(f"已写入: {output_json}")
    print(f"已写入: {output_md}")


if __name__ == "__main__":
    main()

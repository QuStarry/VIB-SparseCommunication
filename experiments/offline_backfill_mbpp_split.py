#!/usr/bin/env python3
"""
离线回填 multi_benchmark JSON 中的 MBPP Early-Stopped / Multi-Round split summary。

当 benchmark JSON 是由旧版 run_multi_benchmark.py 生成（summary 里不带 multi_round_*
key）但 trace_results.full/early_stopped 字段已经存在时，这个脚本会：

  1. 按 trace 中的 early_stopped 字段 + detailed_results 里的消息/token/pass
     数据，离线重新计算 20 个 split key；
  2. 把结果注入 summary[ds_name]；
  3. 同步把 detailed_results 里的每条 row 加 full_early_stopped / vib_early_stopped
     字段，确保新代码回放时一致；
  4. 输出为新的 JSON 文件（默认 *_with_split.json），原文件不修改。
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_pct(numer: float, denom: float) -> float:
    return round((numer / denom) * 100, 2) if denom > 0 else 0.0


def backfill_dataset_split(dataset_name: str, summary: Dict[str, Any],
                           detailed_rows: List[Dict[str, Any]],
                           trace_rows: List[Dict[str, Any]],
                           split_mode: str = "or") -> None:
    """
    split_mode:
      - "or" (default，与主脚本 run_benchmark 第991行的 ES 判断口径保持一致)：
              任一侧 full OR vib early_stopped 就算 ES 子集，只有两侧都没 early_stopped
              才算 MR。MR 样本通常更小，但更"纯多轮"。
      - "and"：两侧 full AND vib 都 early_stopped 才算 ES 子集，
              其余（至少一侧进入多轮）统一归入 Multi-Round。Multi-Round 是更"有压缩
              空间"的"难/需要讨论"的题子集。论文 split 敏感性分析中使用。
    """
    assert split_mode in ("and", "or")
    trace_map = {t["task_id"]: t for t in trace_rows}

    es_count = 0
    mr_count = 0
    es_both_agree = 0
    es_full_passed = 0
    es_vib_passed = 0
    es_full_msgs = 0
    es_vib_msgs = 0
    es_full_tokens = 0
    es_vib_tokens = 0
    es_full_prompt_tokens = 0
    es_vib_prompt_tokens = 0
    es_full_completion_tokens = 0
    es_vib_completion_tokens = 0
    es_same_msg = 0

    mr_full_passed = 0
    mr_vib_passed = 0
    mr_full_msgs = 0
    mr_vib_msgs = 0
    mr_full_tokens = 0
    mr_vib_tokens = 0
    mr_full_prompt_tokens = 0
    mr_vib_prompt_tokens = 0
    mr_full_completion_tokens = 0
    mr_vib_completion_tokens = 0
    mr_same_msg = 0

    for row in detailed_rows:
        tid = row["task_id"]
        trace = trace_map.get(tid, {})
        t_full = trace.get("full", {}) if isinstance(trace, dict) else {}
        t_vib = trace.get("vib", {}) if isinstance(trace, dict) else {}
        f_es = bool(t_full.get("early_stopped"))
        v_es = bool(t_vib.get("early_stopped"))

        row["full_early_stopped"] = f_es
        row["vib_early_stopped"] = v_es

        if split_mode == "and":
            es = f_es and v_es
        else:
            es = f_es or v_es
        f_passed = int(row.get("full_passed", 0))
        v_passed = int(row.get("vib_passed", 0))
        f_msgs = int(row.get("full_messages", 0))
        v_msgs = int(row.get("vib_messages", 0))
        f_tokens = int(row.get("full_tokens", 0))
        v_tokens = int(row.get("vib_tokens", 0))
        f_pt = int(row.get("full_prompt_tokens", 0))
        v_pt = int(row.get("vib_prompt_tokens", 0))
        f_ct = int(row.get("full_completion_tokens", 0))
        v_ct = int(row.get("vib_completion_tokens", 0))
        same = int(f_msgs == v_msgs)

        if f_es and v_es:
            es_both_agree += 1

        if es:
            es_count += 1
            es_full_passed += f_passed
            es_vib_passed += v_passed
            es_full_msgs += f_msgs
            es_vib_msgs += v_msgs
            es_full_tokens += f_tokens
            es_vib_tokens += v_tokens
            es_full_prompt_tokens += f_pt
            es_vib_prompt_tokens += v_pt
            es_full_completion_tokens += f_ct
            es_vib_completion_tokens += v_ct
            es_same_msg += same
        else:
            mr_count += 1
            mr_full_passed += f_passed
            mr_vib_passed += v_passed
            mr_full_msgs += f_msgs
            mr_vib_msgs += v_msgs
            mr_full_tokens += f_tokens
            mr_vib_tokens += v_tokens
            mr_full_prompt_tokens += f_pt
            mr_vib_prompt_tokens += v_pt
            mr_full_completion_tokens += f_ct
            mr_vib_completion_tokens += v_ct
            mr_same_msg += same

    total = summary.get("total_tasks", len(detailed_rows))
    if dataset_name == "MBPP" or es_count > 0:
        es_mc = _safe_pct(es_full_msgs - es_vib_msgs, es_full_msgs)
        mr_mc = _safe_pct(mr_full_msgs - mr_vib_msgs, mr_full_msgs)
        es_ts = _safe_pct(es_full_tokens - es_vib_tokens, es_full_tokens)
        mr_ts = _safe_pct(mr_full_tokens - mr_vib_tokens, mr_full_tokens)

        summary.update({
            "split_mode": split_mode,
            "early_stopped_count": es_count,
            "early_stopped_pct": round(es_count / total * 100, 2) if total > 0 else 0.0,
            "early_stopped_both_agree_count": es_both_agree,
            "multi_round_count": mr_count,
            "multi_round_pct": round(mr_count / total * 100, 2) if total > 0 else 0.0,
            "early_stopped_full_pass_rate": _safe_pct(es_full_passed, es_count),
            "early_stopped_vib_pass_rate": _safe_pct(es_vib_passed, es_count),
            "early_stopped_pass_delta_pp": round(
                _safe_pct(es_vib_passed, es_count) - _safe_pct(es_full_passed, es_count), 2
            ),
            "early_stopped_msg_compression_pct": es_mc,
            "early_stopped_token_saving_pct": es_ts,
            "early_stopped_same_message_ratio": round(es_same_msg / max(1, es_count), 4),
            "multi_round_full_pass_rate": _safe_pct(mr_full_passed, mr_count),
            "multi_round_vib_pass_rate": _safe_pct(mr_vib_passed, mr_count),
            "multi_round_pass_delta_pp": round(
                _safe_pct(mr_vib_passed, mr_count) - _safe_pct(mr_full_passed, mr_count), 2
            ),
            "multi_round_msg_compression_pct": mr_mc,
            "multi_round_token_saving_pct": mr_ts,
            "multi_round_same_message_ratio": round(mr_same_msg / max(1, mr_count), 4),
        })


def main():
    p = argparse.ArgumentParser(description="离线回填 benchmark JSON 的 split summary")
    p.add_argument("--input", required=True, help="输入 benchmark JSON")
    p.add_argument("--output", default=None, help="输出 JSON（默认 *_with_split.json）")
    p.add_argument("--split-mode", choices=("and", "or"), default="or",
                   help="ES=两侧都 early-stop (and) / 任一侧 early-stop (or)，默认 or（与主脚本口径一致）")
    args = p.parse_args()

    in_path = Path(args.input)
    if args.output:
        out_path = Path(args.output)
    else:
        suffix = "_with_split_" + args.split_mode
        out_path = in_path.with_name(in_path.stem + suffix + in_path.suffix)

    report = json.loads(in_path.read_text(encoding="utf-8"))
    report = copy.deepcopy(report)

    summary_bag = report.setdefault("summary", {})
    detailed_bag = report.setdefault("detailed_results", {})
    trace_bag = report.setdefault("trace_results", {})

    for ds_name, rows in detailed_bag.items():
        if not isinstance(rows, list) or not rows:
            continue
        ds_sum = summary_bag.setdefault(ds_name, {})
        ds_trace = trace_bag.get(ds_name, []) or []
        backfill_dataset_split(ds_name, ds_sum, rows, ds_trace, split_mode=args.split_mode)

        print(f"  [OK] mode={args.split_mode} | {ds_name}: "
              f"early_stopped={ds_sum.get('early_stopped_count', 'N/A')}, "
              f"multi_round={ds_sum.get('multi_round_count', 'N/A')}, "
              f"mr_msg_cmp={ds_sum.get('multi_round_msg_compression_pct', 'N/A')}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n回填完成，输出: {out_path}")


if __name__ == "__main__":
    main()

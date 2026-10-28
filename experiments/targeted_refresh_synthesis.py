"""Targeted 重跑 synthesis 阶段（不重跑团队讨论，只替换 generate_final_code）。

用途：P2 修复后，把冻结 baseline 的 HumanEval 164 / MBPP 257 的 synthesis 阶段重新跑一遍。
- 保持原 history 与历史Token计数，不将它们冒充刷新后完整费用
- 替换最终输出和评测标签，并独立保存逐请求刷新账目
- 对数学题 GSM8K 不改动（数学题的 synthesis 是 generate_final_answer，没改）

可选：--resplit-from <trace_refreshed_json> 模式不调用 API，仅在 trace 已刷新的前提下，
对 AND/OR 两种 split_mode 重新调用 offline_backfill_mbpp_split 的核心函数重算 summary。
"""
import argparse, hashlib, importlib.util, json, os, sys, threading
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(__file__))
import run_multi_benchmark as mb
from request_ledger import fingerprint, summarize_requests
from inference_protocol import stage_parameters


_BACKFILL_PATH = os.path.join(os.path.dirname(__file__), "offline_backfill_mbpp_split.py")
_SYNTHESIS_SHA256 = hashlib.sha256(Path(mb.__file__).read_bytes()).hexdigest()


def _load_backfill_module():
    spec = importlib.util.spec_from_file_location("offline_backfill", _BACKFILL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rebuild_dataset_summary_from_refreshed_trace(
    module_ref: Any,
    dataset_name: str,
    traces: List[Dict[str, Any]],
    summary_template: Dict[str, Any],
):
    """根据刷新后的 trace，重新计算 summary 统计（与 MultiDatasetHarness.run_benchmark 字段完全一致）。"""
    tr = module_ref.TaskResult
    total = len(traces)
    task_results = []
    detailed_rows: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "full_passed": 0, "vib_passed": 0,
        "full_tokens": 0, "vib_tokens": 0,
        "full_msgs": 0, "vib_msgs": 0,
        "full_prompt_tokens": 0, "vib_prompt_tokens": 0,
        "full_completion_tokens": 0, "vib_completion_tokens": 0,
        "full_only": 0, "vib_only": 0,
        "both_pass": 0, "both_fail": 0,
        "same_msg_count": 0,
    }
    # -------- PAPER-GAP6 补充字段：ES / MR 自算统计（即使没调 offline_backfill 也能正确渲染） --------
    es_stats: Dict[str, int] = {
        "count": 0, "full_passed": 0, "vib_passed": 0,
        "full_msgs": 0, "vib_msgs": 0,
        "full_tokens": 0, "vib_tokens": 0,
    }
    mr_stats: Dict[str, int] = {
        "count": 0, "full_passed": 0, "vib_passed": 0,
        "full_msgs": 0, "vib_msgs": 0,
        "full_tokens": 0, "vib_tokens": 0,
    }
    # 兼容 split_mode 判定：优先用 trace 自带 early_stopped 标签；否则 fallback 到历史字段（若存在）
    for t in traces:
        fres = t["full"] or {}
        vres = t["vib"] or {}
        fts = fres.get("token_stats", {})
        vts = vres.get("token_stats", {})
        fwf = fres.get("workflow_stats", {})
        vwf = vres.get("workflow_stats", {})
        fmsgs = int(fwf.get("total_approved", 0))
        vmsgs = int(vwf.get("total_approved", 0))
        fp = bool(fres.get("passed"))
        vp = bool(vres.get("passed"))
        f_es = bool(fres.get("early_stopped", False))
        v_es = bool(vres.get("early_stopped", False))
        row = {
            "task_id": t["task_id"],
            "dataset_name": t.get("dataset_name", dataset_name),
            "full_passed": fp,
            "vib_passed": vp,
            "full_messages": fmsgs,
            "vib_messages": vmsgs,
            "full_tokens": int(fts.get("total_tokens", 0)),
            "vib_tokens": int(vts.get("total_tokens", 0)),
            # 让 downstream offline_backfill 不必重复猜测：直接写出每行自身的 early_stopped 标签
            "full_early_stopped": f_es,
            "vib_early_stopped": v_es,
        }
        res = tr(
            task_id=t["task_id"],
            dataset_name=row["dataset_name"],
            full_passed=fp,
            vib_passed=vp,
            full_messages=fmsgs,
            vib_messages=vmsgs,
            full_tokens=row["full_tokens"],
            vib_tokens=row["vib_tokens"],
            execution_time=0.0,
            full_prompt_tokens=int(fts.get("total_prompt_tokens", 0)),
            full_completion_tokens=int(fts.get("total_completion_tokens", 0)),
            vib_prompt_tokens=int(vts.get("total_prompt_tokens", 0)),
            vib_completion_tokens=int(vts.get("total_completion_tokens", 0)),
            full_early_stopped=f_es,
            vib_early_stopped=v_es,
        )
        task_results.append(res)
        detailed_rows.append(row)
        stats["full_passed"] += int(fp)
        stats["vib_passed"] += int(vp)
        stats["full_msgs"] += fmsgs
        stats["vib_msgs"] += vmsgs
        stats["full_tokens"] += res.full_tokens
        stats["vib_tokens"] += res.vib_tokens
        stats["full_prompt_tokens"] += res.full_prompt_tokens
        stats["full_completion_tokens"] += res.full_completion_tokens
        stats["vib_prompt_tokens"] += res.vib_prompt_tokens
        stats["vib_completion_tokens"] += res.vib_completion_tokens
        if fp and not vp:
            stats["full_only"] += 1
        elif vp and not fp:
            stats["vib_only"] += 1
        elif fp and vp:
            stats["both_pass"] += 1
        else:
            stats["both_fail"] += 1
        if fmsgs == vmsgs:
            stats["same_msg_count"] += 1
        # PAPER-GAP6：按 early_stopped 聚合（Full 侧 early_stopped 当作本组判定锚；若两侧不一致，仍只锚定 Full，保持与 offline_backfill 的 OR 语义一致）
        bucket = es_stats if f_es else mr_stats
        bucket["count"] += 1
        bucket["full_passed"] += int(fp)
        bucket["vib_passed"] += int(vp)
        bucket["full_msgs"] += fmsgs
        bucket["vib_msgs"] += vmsgs
        bucket["full_tokens"] += row["full_tokens"]
        bucket["vib_tokens"] += row["vib_tokens"]

    def _safe(numer, denom) -> float:
        return round((numer / denom) * 100, 2) if denom > 0 else 0.0

    def _split_block(prefix: str, s: Dict[str, int], total_count: int) -> Dict[str, Any]:
        n = s["count"]
        return {
            f"{prefix}_count": n,
            f"{prefix}_pct": _safe(n, total_count),
            f"{prefix}_full_pass_rate": _safe(s["full_passed"], n),
            f"{prefix}_vib_pass_rate": _safe(s["vib_passed"], n),
            f"{prefix}_pass_delta_pp": round(
                _safe(s["vib_passed"], n) - _safe(s["full_passed"], n), 2
            ),
            f"{prefix}_msg_compression_pct": _safe(
                s["full_msgs"] - s["vib_msgs"], s["full_msgs"]
            ),
            f"{prefix}_token_saving_pct": _safe(
                s["full_tokens"] - s["vib_tokens"], s["full_tokens"]
            ),
        }

    summary: Dict[str, Any] = {
        "total_tasks": total,
        "full_pass_rate": _safe(stats["full_passed"], total),
        "vib_pass_rate": _safe(stats["vib_passed"], total),
        "pass_delta_pp": round(_safe(stats["vib_passed"], total) - _safe(stats["full_passed"], total), 2),
        "full_messages": stats["full_msgs"],
        "vib_messages": stats["vib_msgs"],
        "msg_compression_pct": _safe(stats["full_msgs"] - stats["vib_msgs"], stats["full_msgs"]),
        "same_message_ratio": round(stats["same_msg_count"] / max(1, total), 4),
        "avg_message_delta": round((stats["full_msgs"] - stats["vib_msgs"]) / max(1, total), 4),
        "full_tokens": stats["full_tokens"],
        "vib_tokens": stats["vib_tokens"],
        "token_saving_pct": _safe(stats["full_tokens"] - stats["vib_tokens"], stats["full_tokens"]),
        "full_prompt_tokens": stats["full_prompt_tokens"],
        "vib_prompt_tokens": stats["vib_prompt_tokens"],
        "prompt_saving_pct": _safe(stats["full_prompt_tokens"] - stats["vib_prompt_tokens"], stats["full_prompt_tokens"]),
        "full_completion_tokens": stats["full_completion_tokens"],
        "vib_completion_tokens": stats["vib_completion_tokens"],
        "completion_saving_pct": _safe(stats["full_completion_tokens"] - stats["vib_completion_tokens"], stats["full_completion_tokens"]),
        "full_only_count": stats["full_only"],
        "vib_only_count": stats["vib_only"],
        "both_pass_count": stats["both_pass"],
        "both_fail_count": stats["both_fail"],
    }
    # PAPER-GAP6：无条件写入 early_stopped_* 与 multi_round_* 两套字段，与主脚本 L1167-L1184 命名完全一致；
    # 后续 offline_backfill 如调用了 _apply_backfill_split，会覆盖更新为 split_mode 精算结果；
    # 未调用时（如 HumanEval / GSM8K）也仍有「按 trace 自带 early_stopped 标签聚合」的默认值。
    summary.update(_split_block("early_stopped", es_stats, total))
    summary.update(_split_block("multi_round", mr_stats, total))
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        import numpy as np
        b = stats["vib_only"]
        c = stats["full_only"]
        table = np.array([[stats["both_pass"], b], [c, stats["both_fail"]]])
        p_raw = mcnemar(table, exact=(b + c) < 25).pvalue
        summary["mcnemar_pvalue"] = round(float(p_raw), 8)
    except Exception:
        # McNemar 为可选增强，缺失依赖时降级为手算 exact 近似（b=c 时 p=1.0）
        b, c = stats["vib_only"], stats["full_only"]
        n = b + c
        if n == 0:
            summary["mcnemar_pvalue"] = 1.0
        else:
            k = min(b, c)
            import math
            p = 0.0
            for i in range(0, k + 1):
                p += math.comb(n, i) * (0.5 ** n)
            summary["mcnemar_pvalue"] = round(min(1.0, 2 * p), 8)
    for k, v in (summary_template or {}).items():
        if k not in summary:
            summary[k] = v
    return summary, task_results, detailed_rows


def _apply_backfill_split(
    module_ref: Any,
    dataset_name: str,
    summary: Dict[str, Any],
    detailed_rows: List[Dict[str, Any]],
    traces: List[Dict[str, Any]],
    split_mode: str,
) -> None:
    backfill_mod = _load_backfill_module()
    backfill_mod.backfill_dataset_split(
        dataset_name, summary, detailed_rows, traces, split_mode=split_mode
    )
    # task_results 同步 detailed_rows 的 early_stopped 字段（便于下游一致性）
    for tr_obj, row in zip(module_ref._last_task_results_list or [], detailed_rows):
        setattr(tr_obj, "full_early_stopped", row.get("full_early_stopped", getattr(tr_obj, "full_early_stopped", False)))
        setattr(tr_obj, "vib_early_stopped", row.get("vib_early_stopped", getattr(tr_obj, "vib_early_stopped", False)))



def _evaluate_code(task_id: str, reference_tests: List[str], code: str, entry_point: str) -> bool:
    """复用 run_multi_benchmark.CodeEvaluator 的子进程 + timeout 沙箱，避免主进程直接 exec。"""
    if not reference_tests:
        return False
    return mb.CodeEvaluator.evaluate(
        code, list(reference_tests), entry_point=entry_point or None, timeout=5.0
    )


def _process_variant(
    variant_trace: Dict[str, Any],
    sample: mb.TaskSample,
    client: Any,
    variant_key: str,
) -> Dict[str, Any]:
    """重新 synthesis 一个 variant（full 或 vib），返回更新后的 variant dict。"""
    output_old = variant_trace.get("output", "")
    passed_old = variant_trace.get("passed", False)
    history = variant_trace.get("history", [])
    prompt = sample.prompt

    if sample.dataset_name == "GSM8K":
        return {**variant_trace, "output": output_old, "passed": passed_old}

    context_hash = fingerprint({
        "task_id": sample.task_id, "dataset": sample.dataset_name, "variant": variant_key,
        "prompt": prompt, "history": history,
        "evaluation": {"tests": sample.test_cases, "entry_point": sample.entry_point},
        "parameters": stage_parameters(client, "code"),
        "model": getattr(client, "model", None), "provider": getattr(client, "provider", None),
        "base_url": getattr(client, "base_url", None), "synthesis_sha256": _SYNTHESIS_SHA256,
    })
    previous = variant_trace.get("refresh_accounting")
    if previous:
        if (previous.get("context_sha256") == context_hash
                and previous.get("output_sha256") == fingerprint(output_old)
                and previous.get("status") == "completed"):
            return variant_trace
        raise ValueError("Existing refresh is unresolved or has different provenance; do not resend automatically")

    if not hasattr(client, "get_request_records"):
        raise ValueError("Refresh requires a request-accounting client")
    request_start = len(client.get_request_records())
    client.request_context = {"task_id": sample.task_id, "variant": variant_key, "purpose": "refresh"}
    output_new = mb.generate_final_code(client, prompt, history)
    requests = client.get_request_records()[request_start:]
    output_hash = fingerprint(output_new)

    passed_new = _evaluate_code(
        sample.task_id, sample.test_cases, output_new, sample.entry_point or ""
    )
    changed = (passed_old != passed_new) or (output_old.strip() != output_new.strip())
    if changed:
        print(
            f"  [{sample.task_id}] {variant_key}: passed {passed_old}→{passed_new} | "
            f"changed={output_old[:60]!r}... != {output_new[:60]!r}..."
        )
    return {
        **variant_trace,
        "output": output_new,
        "passed": passed_new,
        "refresh_accounting": {
            "version": "request_linked_refresh_v1",
            "status": "failed" if not output_new or output_new.startswith("[API Error:") else "completed",
            "context_sha256": context_hash,
            "synthesis_sha256": _SYNTHESIS_SHA256,
            "parent_output_sha256": fingerprint(output_old),
            "output_sha256": output_hash,
            "requests": requests,
            "matching_artifact_request_ids": [
                request["request_id"] for request in requests
                if request.get("artifact_sha256") == output_hash
            ],
            "incremental_usage": summarize_requests(requests),
            "historical_counter_scope": "pre_refresh_only",
            "combined_cost_status": "unknown",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, help="冻结 baseline JSON（API 刷新模式必填）")
    ap.add_argument("--output", required=False, help="输出 synthesis-refreshed 后的新 JSON 路径")
    ap.add_argument("--api-key", default=None, help="可选显式凭证；默认由provider及pro.env选择")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--provider", default=None, choices=["deepseek", "private"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--ledger-dir", default=None, help="逐请求持久化账目目录；默认在输出旁创建")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=-1, help="仅处理前 N 题，用于快速 smoke test")
    ap.add_argument("--resplit-from", required=False,
                    help="不调用 API 的重拆分模式：给定 trace 已刷新的 JSON，对 AND/OR 两种 split_mode 重新计算 summary，并写出 --output-and / --output-or。")
    ap.add_argument("--output-and", required=False, help="--resplit-from 模式下的 AND 划分输出路径")
    ap.add_argument("--output-or", required=False, help="--resplit-from 模式下的 OR 划分输出路径")
    args = ap.parse_args()

    if args.resplit_from:
        _resplit_mode(args)
        return

    if not args.input or not args.output:
        sys.exit("API 刷新模式需要 --input 和 --output")
    in_path = Path(args.input)
    out_path = Path(args.output)
    if in_path.resolve() == out_path.resolve() or out_path.exists():
        sys.exit("必须使用不存在的新输出路径，不能覆盖输入或历史结果")
    ledger_dir = Path(args.ledger_dir) if args.ledger_dir else out_path.with_suffix(".requests")
    if ledger_dir.exists():
        sys.exit("账目目录已存在；先审计pending请求，不自动重发")
    # Resolve and validate before worker threads; no network request is made here.
    try:
        endpoint = mb.build_client(
            args.model, args.provider, temperature=args.temperature, max_tokens=args.max_tokens, log=False,
        )
    except (ValueError, KeyError) as exc:
        sys.exit(f"请求配置无效: {type(exc).__name__}")
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 重建 dataset adapters → TaskSample 映射（拿到 test_cases / prompt / entry_point）
    samples_by_id: Dict[str, mb.TaskSample] = {}
    adapter_map = {
        "humaneval": mb.HumanEvalAdapter,
        "mbpp": mb.MBPPAdapter,
        "gsm8k": mb.GSM8KAdapter,
    }
    for dataset_key, adapter_cls in adapter_map.items():
        try:
            adapter = adapter_cls()
            samples = adapter.load(limit=-1)
            for s in samples:
                samples_by_id[s.task_id] = s
        except Exception as e:
            print(f"[warn] 加载 {dataset_key} 失败，该数据集将跳过 synthesis 刷新: {e}")

    trace_results = data.get("trace_results", {})
    _lock = threading.Lock()
    _summary_changed = {"pass_switches": 0, "total": 0}
    stop_event = threading.Event()
    errors = []

    def _process_dataset(ds_key: str):
        traces: List[Dict[str, Any]] = trace_results.get(ds_key, [])
        client = mb.build_client(
            endpoint.model, endpoint.provider, temperature=args.temperature, max_tokens=args.max_tokens,
            log=False, ledger_path=str(ledger_dir / f"{ds_key}.jsonl"),
        )
        if args.api_key:
            client.api_key = args.api_key
        for idx, trace in enumerate(traces):
            if stop_event.is_set():
                return
            if args.limit > 0 and idx >= args.limit:
                break
            task_id = trace.get("task_id")
            sample = samples_by_id.get(task_id)
            if sample is None:
                continue
            full_old_p = trace["full"].get("passed")
            vib_old_p = trace["vib"].get("passed")
            try:
                new_full = _process_variant(trace["full"], sample, client, "full")
                new_vib = _process_variant(trace["vib"], sample, client, "vib")
            except Exception as e:
                print(f"[error] {task_id}: {type(e).__name__}")
                with _lock:
                    errors.append({"task_id": task_id, "error_type": type(e).__name__})
                stop_event.set()
                return
            trace["full"] = new_full
            trace["vib"] = new_vib
            full_switched = (new_full.get("passed") != full_old_p)
            vib_switched = (new_vib.get("passed") != vib_old_p)
            with _lock:
                _summary_changed["total"] += 1
                if full_switched or vib_switched:
                    _summary_changed["pass_switches"] += 1
                    print(f"  🎯 [{task_id}] pass 变更 F={full_old_p}→{new_full.get('passed')}  "
                          f"V={vib_old_p}→{new_vib.get('passed')}")

    threads = []
    for ds in ("humaneval", "mbpp", "gsm8k"):
        if ds in trace_results:
            t = threading.Thread(target=_process_dataset, args=(ds,), daemon=True)
            threads.append(t)
            t.start()
    for t in threads:
        t.join()
    if errors:
        sys.exit("刷新已停止；保留请求账目，不生成可被误认为完整的结果文件")

    # 刷新 summary / detailed_results（与 refreshed trace.passed 重新对齐，消息/Token 保持不变）
    new_summary: Dict[str, Any] = {}
    new_detailed: Dict[str, Any] = {}
    last_task_results: List[Any] = []
    mb._last_task_results_list = last_task_results
    try:
        for ds in ("humaneval", "mbpp", "gsm8k"):
            traces = trace_results.get(ds) or []
            if not traces:
                continue
            tpl = data.get("summary", {}).get(ds) or {}
            summary, task_results, detailed_rows = _rebuild_dataset_summary_from_refreshed_trace(
                mb, ds, traces, tpl
            )
            last_task_results.clear()
            last_task_results.extend(task_results)
            if ds == "mbpp":
                _apply_backfill_split(mb, "MBPP", summary, detailed_rows, traces, split_mode="or")
            new_summary[ds] = summary
            new_detailed[ds] = detailed_rows
    finally:
        try:
            delattr(mb, "_last_task_results_list")
        except AttributeError:
            pass

    data["summary"] = new_summary
    data["detailed_results"] = new_detailed
    data["refreshed_synthesis"] = True
    data["refreshed_synthesis_source"] = str(args.input)
    data["refreshed_summary"] = _summary_changed.copy()
    data["refreshed_cost_status"] = "unknown"
    data["refreshed_token_counter_scope"] = "pre_refresh_only"
    data["request_ledger_directory"] = str(ledger_dir)
    print(f"[done] synthesis 刷新题数: {_summary_changed['total']}，其中 pass 变更: {_summary_changed['pass_switches']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[saved] → {out_path}")


def _resplit_mode(args):
    if not args.output_and or not args.output_or:
        sys.exit("--resplit-from 模式需要同时指定 --output-and 和 --output-or")
    with open(args.resplit_from, "r", encoding="utf-8") as f:
        data = json.load(f)
    trace_results = data.get("trace_results", {})
    base_summary = data.get("summary", {})

    new_summaries = {"and": {}, "or": {}}
    new_detailed = {"and": {}, "or": {}}
    last_task_results: List[Any] = []
    class _HackRef:
        pass
    mb._last_task_results_list = last_task_results  # 临时挂载，供 _apply_backfill_split 同步
    try:
        for ds in ("humaneval", "mbpp", "gsm8k"):
            traces = trace_results.get(ds) or []
            if not traces:
                continue
            tpl = base_summary.get(ds) or {}
            summary, task_results, detailed_rows = _rebuild_dataset_summary_from_refreshed_trace(
                mb, ds, traces, tpl
            )
            last_task_results.clear()
            last_task_results.extend(task_results)
            for mode in ("and", "or"):
                sum_mode = dict(summary)
                rows_mode = [dict(r) for r in detailed_rows]
                if ds == "mbpp":
                    _apply_backfill_split(mb, "MBPP", sum_mode, rows_mode, traces, split_mode=mode)
                new_summaries[mode][ds] = sum_mode
                new_detailed[mode][ds] = rows_mode
    finally:
        try:
            delattr(mb, "_last_task_results_list")
        except AttributeError:
            pass

    for mode, out_path in (("and", args.output_and), ("or", args.output_or)):
        out_data = dict(data)
        out_data["summary"] = new_summaries[mode]
        out_data["detailed_results"] = new_detailed[mode]
        out_data["refreshed_resplit_mode"] = mode
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"[saved resplit {mode}] → {out_path}")


if __name__ == "__main__":
    main()

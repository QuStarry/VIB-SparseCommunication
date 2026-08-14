"""
Vanilla (Single-Agent, No Multi-Agent Discussion) Baseline Runner.

Gap 5 — 顶会贡献三列对比最左列：Vanilla / Full-Comm (3-Agent) / VIB-Sparse (3-Agent + VIB).

为了 100% 复用 analyze_multi_benchmark / generate_report / offline_backfill_mbpp_split 的
现有管道（这些脚本按 full/vib 双列写死），我们把「单 agent 直接 synthesis 结果」同时写
入 full 与 vib 两侧（两侧同值）。语义上：
- pass_rate: vanilla_pass_rate == full_pass_rate == vib_pass_rate
- msg_compression_pct: 0.00%（没有讨论，不存在压缩的概念；保留分母用 1 条 synthesis 消息）
- token_saving_pct: 0.00%

MBPP ES probe 规则：直接 synthesis 若 Pass 则记 EarlyStopped=True，否则视为 MR（逻辑与主脚本一致）。

Usage:
    python3 experiments/run_vanilla_baseline.py --datasets humaneval mbpp gsm8k --limit 3   # dry-run
    python3 experiments/run_vanilla_baseline.py --datasets humaneval mbpp gsm8k --output results/vanilla_baseline_3dataset.json
    python3 experiments/run_vanilla_baseline.py --workers 4
"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import json
import argparse
import threading
from dataclasses import asdict
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import run_multi_benchmark as mb


class VanillaBaselineHarness:
    """单 agent baseline：直接跳过讨论阶段，synthesis 端到端生成 final code/answer。"""

    def __init__(self, api_key: str, workers: int = 1,
                 experiment_config: mb.ExperimentConfig = None):
        self.api_key = api_key
        self.workers = max(1, workers)
        self._print_lock = threading.Lock()
        self.experiment_config = experiment_config or mb.ExperimentConfig()

    # ------------------------------------------------------------------
    # Core executor
    # ------------------------------------------------------------------
    def _select_components(self, sample: mb.TaskSample) -> Tuple[Any, Any]:
        # 与主脚本保持一致的路由；history 传空 list 相当于「零讨论、直接 synthesis」
        if sample.dataset_name in ("HumanEval", "MBPP"):
            return mb.create_code_agents, mb.generate_final_code
        return mb.create_math_agents, mb.generate_final_answer

    def _run_vanilla_variant(self, sample: mb.TaskSample) -> Dict[str, Any]:
        """运行 vanilla 单分支：直接 synthesis，然后 Evaluator 打标。

        同时返回用于 trace 的 payload 字段。
        """
        _, generate_fn = self._select_components(sample)
        # 单 agent，用 synthesis 的配置（temperature=0.1，max_tokens=1024）——与主脚本 synthesis
        # 保持相同参数，保证 apples-to-apples 对比。
        client = mb.DeepSeekClient(api_key=self.api_key, max_tokens=1024)
        start = time.time()
        final_output = generate_fn(client, sample.prompt, history=[])
        exec_time = time.time() - start
        passed = mb.UnifiedEvaluator.evaluate(sample, final_output)
        token_stats = client.get_token_stats()
        total_tokens = int(token_stats.get("total_tokens", 0))
        total_prompt = int(token_stats.get("total_prompt_tokens", 0))
        total_completion = int(token_stats.get("total_completion_tokens", 0))
        # 人为规定：单 agent 对外暴露 total_approved=1 条「最终输出消息」，避免 0 做分母
        workflow_stats = {
            "total_approved": 1,
            "total_proposed": 1,
            "steps_run": 1,
            "early_terminated": False,
        }
        return {
            "history": [],  # 无讨论
            "workflow_stats": workflow_stats,
            "output": final_output,
            "passed": passed,
            "token_stats": {
                "total_tokens": total_tokens,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
            },
            "execution_time": exec_time,
        }

    def _build_trace_payload(self, sample: mb.TaskSample, variant_name: str,
                             variant_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "variant": variant_name,
            "task_id": sample.task_id,
            "dataset_name": sample.dataset_name,
            "prompt": sample.prompt,
            "reference_answer": sample.reference_answer,
            "entry_point": sample.entry_point,
            "test_cases": sample.test_cases,
            "metadata": sample.metadata,
            "passed": variant_result["passed"],
            "history": variant_result["history"],
            "output": variant_result["output"],
            "workflow_stats": variant_result["workflow_stats"],
            "token_stats": variant_result["token_stats"],
            "early_stopped": variant_result.get("early_stopped", False),
        }

    def run_single_task_with_trace(self, sample: mb.TaskSample) -> Dict[str, Any]:
        task_start = time.time()
        try:
            # MBPP：按主脚本 ES 协议——如果直接 synthesis 直接过，则记 ES（与主脚本 probe 语义一致）
            base_result = self._run_vanilla_variant(sample)
            f_es = False
            if sample.dataset_name == "MBPP" and self.experiment_config.enable_mbpp_early_stop:
                if base_result["passed"]:
                    f_es = True
            base_result["early_stopped"] = f_es

            token_stats = base_result["token_stats"]
            wf_stats = base_result["workflow_stats"]

            # 为了复用主脚本 analyze/generate_report，两侧 full/vib 写入完全相同的 vanilla 结果
            full_result = dict(base_result)
            vib_result = dict(base_result)

            task_result = mb.TaskResult(
                task_id=sample.task_id,
                dataset_name=sample.dataset_name,
                full_passed=full_result["passed"],
                vib_passed=vib_result["passed"],
                full_messages=wf_stats["total_approved"],
                vib_messages=wf_stats["total_approved"],
                full_tokens=token_stats["total_tokens"],
                vib_tokens=token_stats["total_tokens"],
                execution_time=time.time() - task_start,
                full_prompt_tokens=token_stats["total_prompt_tokens"],
                full_completion_tokens=token_stats["total_completion_tokens"],
                vib_prompt_tokens=token_stats["total_prompt_tokens"],
                vib_completion_tokens=token_stats["total_completion_tokens"],
                full_early_stopped=f_es,
                vib_early_stopped=f_es,
            )
            return {
                "task_result": task_result,
                "full": full_result,
                "vib": vib_result,
            }
        except Exception as e:
            exec_time = time.time() - task_start
            err = mb.TaskResult(
                task_id=sample.task_id,
                dataset_name=sample.dataset_name,
                full_passed=False, vib_passed=False,
                full_messages=0, vib_messages=0,
                full_tokens=0, vib_tokens=0,
                execution_time=exec_time,
                error_msg=str(e),
            )
            return {"task_result": err, "full": None, "vib": None}

    # ------------------------------------------------------------------
    # Aggregation & entry
    # ------------------------------------------------------------------
    def run_benchmark(self, datasets: List[str], limit: int = -1) -> Dict[str, Any]:
        all_results: Dict[str, Any] = {}
        trace_results: Dict[str, Any] = {}
        summary: Dict[str, Any] = {}
        global_start = time.time()

        for ds_name in datasets:
            print(f"\n{'='*60}")
            print(f" 🧪 Vanilla Baseline: {ds_name.upper()} (limit={limit if limit > 0 else 'ALL'})")
            print(f"{'='*60}")

            adapter = mb.DatasetFactory.get_adapter(ds_name)
            samples = adapter.load(limit=limit)
            if not samples:
                print(f"  [Skipped] 无数据加载: {ds_name}")
                continue
            total = len(samples)
            print(f"  已加载 {total} 个样本，workers={self.workers}")

            results: List[Any] = [None] * total
            traces: List[Any] = [None] * total
            stats: Dict[str, int] = {
                "full_passed": 0, "vib_passed": 0,
                "full_tokens": 0, "vib_tokens": 0,
                "full_msgs": 0, "vib_msgs": 0,
                "full_prompt_tokens": 0, "vib_prompt_tokens": 0,
                "full_completion_tokens": 0, "vib_completion_tokens": 0,
                "full_only": 0, "vib_only": 0,
                "both_pass": 0, "both_fail": 0,
                "same_msg_count": 0,
                "early_stop_any": 0, "early_stop_both": 0,
                "es_full_passed": 0, "es_vib_passed": 0,
                "es_full_msgs": 0, "es_vib_msgs": 0,
                "es_full_tokens": 0, "es_vib_tokens": 0,
                "es_full_prompt_tokens": 0, "es_vib_prompt_tokens": 0,
                "es_full_completion_tokens": 0, "es_vib_completion_tokens": 0,
                "es_same_msg_count": 0,
                "mr_total": 0,
                "mr_full_passed": 0, "mr_vib_passed": 0,
                "mr_full_msgs": 0, "mr_vib_msgs": 0,
                "mr_full_tokens": 0, "mr_vib_tokens": 0,
                "mr_full_prompt_tokens": 0, "mr_vib_prompt_tokens": 0,
                "mr_full_completion_tokens": 0, "mr_vib_completion_tokens": 0,
                "mr_same_msg_count": 0,
            }
            dataset_start = time.time()

            def consume(idx: int, bundle: Dict[str, Any], done: int):
                res = bundle["task_result"]
                results[idx] = res
                sample = samples[idx]
                traces[idx] = {
                    "task_id": sample.task_id,
                    "dataset_name": sample.dataset_name,
                    "full": self._build_trace_payload(sample, "full", bundle["full"]) if bundle.get("full") else None,
                    "vib": self._build_trace_payload(sample, "vib", bundle["vib"]) if bundle.get("vib") else None,
                    "error_msg": res.error_msg,
                }
                stats["full_passed"] += int(res.full_passed)
                stats["vib_passed"] += int(res.vib_passed)
                stats["full_tokens"] += res.full_tokens
                stats["vib_tokens"] += res.vib_tokens
                stats["full_msgs"] += res.full_messages
                stats["vib_msgs"] += res.vib_messages
                stats["full_prompt_tokens"] += res.full_prompt_tokens
                stats["vib_prompt_tokens"] += res.vib_prompt_tokens
                stats["full_completion_tokens"] += res.full_completion_tokens
                stats["vib_completion_tokens"] += res.vib_completion_tokens
                stats["same_msg_count"] += int(res.full_messages == res.vib_messages)

                es = bool(res.full_early_stopped or res.vib_early_stopped)
                if es:
                    stats["early_stop_any"] += 1
                    stats["es_full_passed"] += int(res.full_passed)
                    stats["es_vib_passed"] += int(res.vib_passed)
                    stats["es_full_msgs"] += res.full_messages
                    stats["es_vib_msgs"] += res.vib_messages
                    stats["es_full_tokens"] += res.full_tokens
                    stats["es_vib_tokens"] += res.vib_tokens
                    stats["es_full_prompt_tokens"] += res.full_prompt_tokens
                    stats["es_vib_prompt_tokens"] += res.vib_prompt_tokens
                    stats["es_full_completion_tokens"] += res.full_completion_tokens
                    stats["es_vib_completion_tokens"] += res.vib_completion_tokens
                    stats["es_same_msg_count"] += int(res.full_messages == res.vib_messages)
                    if res.full_early_stopped and res.vib_early_stopped:
                        stats["early_stop_both"] += 1
                else:
                    stats["mr_total"] += 1
                    stats["mr_full_passed"] += int(res.full_passed)
                    stats["mr_vib_passed"] += int(res.vib_passed)
                    stats["mr_full_msgs"] += res.full_messages
                    stats["mr_vib_msgs"] += res.vib_messages
                    stats["mr_full_tokens"] += res.full_tokens
                    stats["mr_vib_tokens"] += res.vib_tokens
                    stats["mr_full_prompt_tokens"] += res.full_prompt_tokens
                    stats["mr_vib_prompt_tokens"] += res.vib_prompt_tokens
                    stats["mr_full_completion_tokens"] += res.full_completion_tokens
                    stats["mr_vib_completion_tokens"] += res.vib_completion_tokens
                    stats["mr_same_msg_count"] += int(res.full_messages == res.vib_messages)

                if res.full_passed and res.vib_passed:
                    stats["both_pass"] += 1
                elif res.full_passed and not res.vib_passed:
                    stats["full_only"] += 1
                elif not res.full_passed and res.vib_passed:
                    stats["vib_only"] += 1
                else:
                    stats["both_fail"] += 1

                elapsed = time.time() - dataset_start
                avg = elapsed / max(1, done)
                eta = avg * (total - done) / 60.0
                icon = "✅" if res.full_passed else "❌"
                with self._print_lock:
                    print(f"  [{done:03d}/{total:03d}] {res.task_id:<25} | "
                          f"Vanilla: {icon} (tokens {res.full_tokens}) | "
                          f"耗时: {res.execution_time:.1f}s | ETA: {eta:.1f}min")

            if self.workers == 1:
                for idx, sample in enumerate(samples, start=1):
                    consume(idx - 1, self.run_single_task_with_trace(sample), idx)
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as ex:
                    fmap = {ex.submit(self.run_single_task_with_trace, s): i
                            for i, s in enumerate(samples)}
                    done = 0
                    for f in as_completed(fmap):
                        done += 1
                        consume(fmap[f], f.result(), done)

            # 汇总（字段名与 run_multi_benchmark.py:L1130-L1184 一字不差）
            def _sp(n: int, d: int) -> float:
                return round((n / d) * 100, 2) if d > 0 else 0.0

            full_rate = _sp(stats["full_passed"], total)
            vib_rate = _sp(stats["vib_passed"], total)
            summary[ds_name] = {
                "total_tasks": total,
                "full_pass_rate": full_rate,
                "vib_pass_rate": vib_rate,
                "pass_delta_pp": round(vib_rate - full_rate, 2),
                "full_messages": stats["full_msgs"],
                "vib_messages": stats["vib_msgs"],
                "msg_compression_pct": _sp(stats["full_msgs"] - stats["vib_msgs"], stats["full_msgs"]),
                "same_message_ratio": round(stats["same_msg_count"] / max(1, total), 4),
                "avg_message_delta": round((stats["full_msgs"] - stats["vib_msgs"]) / max(1, total), 4),
                "full_tokens": stats["full_tokens"],
                "vib_tokens": stats["vib_tokens"],
                "token_saving_pct": _sp(stats["full_tokens"] - stats["vib_tokens"], stats["full_tokens"]),
                "full_prompt_tokens": stats["full_prompt_tokens"],
                "vib_prompt_tokens": stats["vib_prompt_tokens"],
                "prompt_saving_pct": _sp(stats["full_prompt_tokens"] - stats["vib_prompt_tokens"], stats["full_prompt_tokens"]),
                "full_completion_tokens": stats["full_completion_tokens"],
                "vib_completion_tokens": stats["vib_completion_tokens"],
                "completion_saving_pct": _sp(stats["full_completion_tokens"] - stats["vib_completion_tokens"], stats["full_completion_tokens"]),
                "full_only_count": stats["full_only"],
                "vib_only_count": stats["vib_only"],
                "both_pass_count": stats["both_pass"],
                "both_fail_count": stats["both_fail"],
                "split_mode": "or" if ds_name == "mbpp" or ds_name.upper() == "MBPP" else "n/a",
            }
            # 补 McNemar pvalue（与 targeted_refresh 同实现；优先 statsmodels，否则 exact 近似）——因为
            # vanilla full/vib 同值，b=c=0，pvalue 应为 1.0，与语义一致。
            b, c = stats["vib_only"], stats["full_only"]
            n = b + c
            try:
                from statsmodels.stats.contingency_tables import mcnemar
                import numpy as np
                table = np.array([[stats["both_pass"], b], [c, stats["both_fail"]]])
                p_raw = mcnemar(table, exact=(b + c) < 25).pvalue
                summary[ds_name]["mcnemar_pvalue"] = round(float(p_raw), 8)
            except Exception:
                if n == 0:
                    summary[ds_name]["mcnemar_pvalue"] = 1.0
                else:
                    import math
                    k = min(b, c)
                    p = 0.0
                    for i in range(0, k + 1):
                        p += math.comb(n, i) * (0.5 ** n)
                    summary[ds_name]["mcnemar_pvalue"] = round(min(1.0, 2 * p), 8)
            if ds_name == "MBPP" or stats["early_stop_any"] > 0:
                es_n = stats["early_stop_any"]
                mr_n = stats["mr_total"]
                summary[ds_name].update({
                    "early_stopped_count": es_n,
                    "early_stopped_pct": round(es_n / total * 100, 2),
                    "early_stopped_both_agree_count": stats["early_stop_both"],
                    "multi_round_count": mr_n,
                    "multi_round_pct": round(mr_n / total * 100, 2),
                    "early_stopped_full_pass_rate": _sp(stats["es_full_passed"], es_n),
                    "early_stopped_vib_pass_rate": _sp(stats["es_vib_passed"], es_n),
                    "early_stopped_pass_delta_pp": round(
                        _sp(stats["es_vib_passed"], es_n) - _sp(stats["es_full_passed"], es_n), 2
                    ),
                    "early_stopped_msg_compression_pct": _sp(
                        stats["es_full_msgs"] - stats["es_vib_msgs"], stats["es_full_msgs"]
                    ),
                    "early_stopped_token_saving_pct": _sp(
                        stats["es_full_tokens"] - stats["es_vib_tokens"], stats["es_full_tokens"]
                    ),
                    "early_stopped_same_message_ratio": round(stats["es_same_msg_count"] / max(1, es_n), 4),
                    "multi_round_full_pass_rate": _sp(stats["mr_full_passed"], mr_n),
                    "multi_round_vib_pass_rate": _sp(stats["mr_vib_passed"], mr_n),
                    "multi_round_pass_delta_pp": round(
                        _sp(stats["mr_vib_passed"], mr_n) - _sp(stats["mr_full_passed"], mr_n), 2
                    ),
                    "multi_round_msg_compression_pct": _sp(
                        stats["mr_full_msgs"] - stats["mr_vib_msgs"], stats["mr_full_msgs"]
                    ),
                    "multi_round_token_saving_pct": _sp(
                        stats["mr_full_tokens"] - stats["mr_vib_tokens"], stats["mr_full_tokens"]
                    ),
                    "multi_round_same_message_ratio": round(stats["mr_same_msg_count"] / max(1, mr_n), 4),
                })

            all_results[ds_name] = [asdict(r) for r in results if r is not None]
            trace_results[ds_name] = [t for t in traces if t is not None]

        total_time = (time.time() - global_start) / 60.0
        print(f"\nVanilla 评测总耗时: {total_time:.2f} 分钟")
        self._print_report(summary)

        return {
            "summary": summary,
            "detailed_results": all_results,
            "trace_results": trace_results,
            "experiment_config": asdict(self.experiment_config),
            "meta": {"mode": "vanilla_single_agent_synthesis_direct",
                     "description": "full/vib 双列写入同一 vanilla 结果，便于复用 analyze/generate_report"},
        }

    def _print_report(self, summary: Dict[str, Any]):
        print(f"\n{'='*80}")
        print(" 📊 Vanilla Baseline Report (single-agent direct synthesis)")
        print(f"{'='*80}")
        print(f"{'Dataset':<12} | {'Pass@1':>10} | {'Tokens/task':>12} | {'Msgs/task':>11}")
        print("-" * 60)
        for ds, m in summary.items():
            tt = m["total_tasks"] or 1
            print(f"{ds:<12} | {m['full_pass_rate']:>8.2f}% | "
                  f"{m['full_tokens'] / tt:>10.1f} | "
                  f"{m['full_messages'] / tt:>9.2f}")


def main():
    parser = argparse.ArgumentParser(description="Vanilla (single-agent direct synthesis) baseline runner")
    parser.add_argument("--datasets", nargs="+", default=["humaneval", "mbpp", "gsm8k"],
                        help="数据集列表 (humaneval, mbpp, gsm8k, math)")
    parser.add_argument("--limit", type=int, default=-1, help="每数据集 limit 样本（<=0 表示全量）")
    parser.add_argument("--workers", type=int, default=1, help="并行线程数")
    parser.add_argument("--output", default="results/vanilla_baseline_3dataset.json",
                        help="输出 JSON 路径")
    parser.add_argument("--api-key", default=None, help="可选；否则读 env DEEPSEEK_API_KEY / .env")
    args = parser.parse_args()

    api_key = args.api_key or mb.resolve_deepseek_api_key()
    if not api_key:
        print("❌ 找不到 DeepSeek API Key。请设置 DEEPSEEK_API_KEY 或在 .env 中配置。")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cfg = mb.ExperimentConfig()
    harness = VanillaBaselineHarness(api_key=api_key, workers=args.workers, experiment_config=cfg)
    result = harness.run_benchmark(args.datasets, limit=args.limit)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已写盘: {args.output}")
    print(f"   数据集: {list(result['summary'].keys())}")
    for ds, s in result["summary"].items():
        print(f"   - {ds}: Pass@1 {s['full_pass_rate']:.2f}% (n={s['total_tasks']})")


if __name__ == "__main__":
    main()

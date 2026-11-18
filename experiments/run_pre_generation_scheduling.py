#!/usr/bin/env python3
"""Pre-generation scheduling benchmark driver.

Implements the frozen protocol in
``paper/pre_generation_scheduling_protocol_zh.md``: for every task it runs a
fixed set of conditions and records, per condition, the pass/fail outcome and
the three communication cohorts (broadcast messages, prompt tokens, total
tokens). The point of the experiment is that a skipped role is *never called*,
so the tokens it would have spent are saved rather than merely re-budgeted.

Conditions
----------
full         BypassVIBRouter, no scheduler. Upper bound: every role runs and
             every candidate is broadcast.
post_gate    VIB-Sparse gate applied *after* generation, no scheduler. This is
             the method under study today; generation cost is a sunk cost.
fixed_trim   VIB gate + a fixed round horizon (round 1 only). A trivial
             trimming baseline.
random_skip  VIB gate + count-matched random control: each round it keeps the
             same number of roles as Coverage but picks them uniformly at
             random (seeded per task and round). Isolates the scheduling signal
             (which roles) from the mere reduction in calls (how many).
coverage     VIB gate + role-coverage scheduler: later rounds run only roles
             whose messages never reached the shared history, plus a mandatory
             closer role.

All schedulers are pure functions of evidence visible at decision time (goal,
approved history, round index, role names). None of them consult reference
answers.

Scope guard: this driver will not run for HumanEval/MATH unless the historical
task ids are excluded, so a "fresh task" claim cannot silently reuse tasks that
were already spent in earlier experiments.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "..", "src")
for _p in (_THIS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_multi_benchmark import (  # noqa: E402
    BypassVIBRouter,
    DatasetFactory,
    ExperimentConfig,
    InferenceProtocol,
    MultiDatasetHarness,
    TaskSample,
    VIBSparseCommRouter,
    build_client,
)
from pre_generation_scheduling import (  # noqa: E402
    count_matched_random_scheduler,
    fixed_round_scheduler,
    role_coverage_scheduler,
)

ALL_CONDITIONS: Tuple[str, ...] = ("full", "post_gate", "fixed_trim", "random_skip", "coverage")

# The role that must always be allowed to run in the coverage policy: the one
# whose artefact the final answer is synthesised from.
CLOSER_ROLES: Dict[str, Tuple[str, ...]] = {
    "HumanEval": ("Coder",),
    "MBPP": ("Coder",),
    "GSM8K": ("Solver",),
    "MATH": ("Solver",),
}

# Datasets whose historical task ids must not be reused without an explicit
# acknowledgement flag.
HISTORY_SENSITIVE_DATASETS: Tuple[str, ...] = ("HumanEval", "MBPP", "GSM8K", "MATH")


# ---------------------------------------------------------------------------
# Pure logic (unit tested without any model calls)
# ---------------------------------------------------------------------------

def build_condition(
    name: str,
    dataset_name: str,
    harness: MultiDatasetHarness,
    task_id: str,
    random_seed: int,
) -> Tuple[Any, Optional[Any]]:
    """Return the ``(router, scheduler)`` pair for a condition.

    Only ``full`` bypasses the gate; every other condition keeps the VIB gate on
    so the *only* factor that separates ``post_gate`` / ``fixed_trim`` /
    ``random_skip`` / ``coverage`` is the pre-generation scheduler.
    """
    if name == "full":
        return BypassVIBRouter(), None

    router = harness._create_vib_router(task_id)
    if name == "post_gate":
        return router, None
    if name == "fixed_trim":
        return router, fixed_round_scheduler(max_rounds=1)

    coverage_policy = role_coverage_scheduler(closer_roles=CLOSER_ROLES.get(dataset_name, ()))
    if name == "coverage":
        return router, coverage_policy
    if name == "random_skip":
        # Count-matched control: same number of roles as Coverage each round,
        # but chosen uniformly at random instead of by history coverage.
        return router, count_matched_random_scheduler(
            base_scheduler=coverage_policy, seed=random_seed, task_salt=task_id
        )
    raise ValueError(f"未知条件: {name}")


def paired_bootstrap_ci(
    diffs: Sequence[float], n_boot: int = 10000, seed: int = 20260812
) -> Optional[Dict[str, float]]:
    """Percentile bootstrap CI for the mean of paired per-task differences."""
    values = [float(d) for d in diffs]
    if not values:
        return None
    rng = random.Random(seed)
    m = len(values)
    means: List[float] = []
    for _ in range(n_boot):
        acc = 0.0
        for _ in range(m):
            acc += values[rng.randrange(m)]
        means.append(acc / m)
    means.sort()

    def _pct(p: float) -> float:
        idx = min(len(means) - 1, max(0, int(p * len(means))))
        return means[idx]

    return {
        "n": m,
        "n_boot": n_boot,
        "mean": sum(values) / m,
        "lo": _pct(0.025),
        "hi": _pct(0.975),
    }


def summarize_conditions(
    records: Sequence[Dict[str, Any]], conditions: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Aggregate the per-task records into per-condition totals."""
    summary: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        rows = [r["conditions"][cond] for r in records if cond in r["conditions"]]
        n = len(rows)
        passed = sum(int(bool(r.get("passed"))) for r in rows)
        summary[cond] = {
            "n": n,
            "passed": passed,
            "accuracy_pct": (100.0 * passed / n) if n else 0.0,
            "messages": sum(int(r.get("messages", 0)) for r in rows),
            "candidates": sum(int(r.get("candidates", 0)) for r in rows),
            "skipped_roles": sum(int(r.get("skipped_roles", 0)) for r in rows),
            "prompt_tokens": sum(int(r.get("prompt_tokens", 0)) for r in rows),
            "completion_tokens": sum(int(r.get("completion_tokens", 0)) for r in rows),
            "total_tokens": sum(int(r.get("total_tokens", 0)) for r in rows),
            "errors": sum(1 for r in rows if r.get("error")),
        }
    return summary


def compare_conditions(
    records: Sequence[Dict[str, Any]],
    cond_a: str,
    cond_b: str,
    summary: Dict[str, Dict[str, Any]],
    n_boot: int = 10000,
    seed: int = 20260812,
) -> Dict[str, Any]:
    """``cond_a - cond_b``: paired accuracy delta plus aggregate reductions."""
    pairs = [
        (r["conditions"][cond_a], r["conditions"][cond_b])
        for r in records
        if cond_a in r["conditions"] and cond_b in r["conditions"]
    ]
    acc_diffs = [100.0 * (int(bool(a.get("passed"))) - int(bool(b.get("passed")))) for a, b in pairs]

    def _reduction_pct(key: str) -> Optional[float]:
        base = summary.get(cond_b, {}).get(key)
        if not base:
            return None
        return 100.0 * (base - summary.get(cond_a, {}).get(key, 0)) / base

    return {
        "a": cond_a,
        "b": cond_b,
        "n_pairs": len(pairs),
        "accuracy_delta_pp": paired_bootstrap_ci(acc_diffs, n_boot=n_boot, seed=seed),
        "reduction_pct": {
            "messages": _reduction_pct("messages"),
            "prompt_tokens": _reduction_pct("prompt_tokens"),
            "total_tokens": _reduction_pct("total_tokens"),
        },
    }


# ---------------------------------------------------------------------------
# Historical-task exclusion
# ---------------------------------------------------------------------------

def collect_task_ids(obj: Any, sink: Set[str]) -> None:
    """Recursively collect every ``task_id`` from a results manifest.

    Walks *all* nested dicts and lists rather than a fixed key whitelist: some
    result files keep their per-task payload under keys such as
    ``detailed_results`` or ``trace_results``, and missing those would silently
    mark already-spent tasks as fresh. Over-collecting is safe here because
    ``sink`` feeds an exclusion list; under-collecting is not.
    """
    if isinstance(obj, dict):
        task_id = obj.get("task_id")
        if isinstance(task_id, str):
            sink.add(task_id)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                collect_task_ids(value, sink)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                sink.add(item)
            elif isinstance(item, (dict, list)):
                collect_task_ids(item, sink)
    elif isinstance(obj, str):
        sink.add(obj)


def load_exclusions(manifest_paths: Iterable[str], inline_ids: Iterable[str]) -> Set[str]:
    excluded: Set[str] = {str(i) for i in inline_ids}
    for raw_path in manifest_paths or []:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"exclude manifest 不存在: {path}")
        with path.open("r", encoding="utf-8") as handle:
            collect_task_ids(json.load(handle), excluded)
    return excluded


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _record_for_condition(variant: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
    if error is not None:
        return {
            "passed": False,
            "messages": 0,
            "candidates": 0,
            "skipped_roles": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": error,
        }
    stats = variant.get("workflow_stats", {}) or {}
    tokens = variant.get("token_stats", {}) or {}
    decisions = stats.get("schedule_decisions", []) or []
    return {
        "passed": bool(variant.get("passed")),
        "messages": int(stats.get("total_approved", 0)),
        "candidates": int(stats.get("total_candidates", 0)),
        "skipped_roles": sum(len(d.get("skipped", [])) for d in decisions),
        "prompt_tokens": int(tokens.get("total_prompt_tokens", 0)),
        "completion_tokens": int(tokens.get("total_completion_tokens", 0)),
        "total_tokens": int(tokens.get("total_tokens", 0)),
        "error": None,
    }


def run_task(
    harness: MultiDatasetHarness,
    sample: TaskSample,
    conditions: Sequence[str],
    random_seed: int,
) -> Dict[str, Any]:
    create_agents_fn, generate_fn = harness._select_task_components(sample)
    record: Dict[str, Any] = {
        "task_id": sample.task_id,
        "dataset": sample.dataset_name,
        "conditions": {},
    }
    for cond in conditions:
        try:
            router, scheduler = build_condition(
                cond,
                sample.dataset_name,
                harness,
                sample.task_id,
                random_seed,
            )
            variant = harness._run_variant(
                sample=sample,
                create_agents_fn=create_agents_fn,
                generate_fn=generate_fn,
                router=router,
                pre_generation_scheduler=scheduler,
                variant_label=cond,
            )
            record["conditions"][cond] = _record_for_condition(variant)
        except Exception as exc:  # pragma: no cover - defensive, exercised online
            record["conditions"][cond] = _record_for_condition({}, error=f"{type(exc).__name__}: {exc}")
    return record


def _task_id_sort_key(task_id: str):
    """按 ``前缀/序号`` 的自然顺序排序，避免字典序把 ``/10`` 排到 ``/2`` 之前。

    这样 ``GSM8K-train/0..299`` 才与协议 §6 的题号区间逐字一致；非数字后缀
    退化为字符串序，保证任意 task_id 都可排序。
    """
    prefix, _, suffix = task_id.rpartition("/")
    try:
        return (prefix, 0, int(suffix))
    except ValueError:
        return (prefix, 1, suffix)


def _select_samples(dataset: str, limit: int, excluded: Set[str]) -> List[TaskSample]:
    adapter = DatasetFactory.get_adapter(dataset)
    samples = adapter.load(limit=-1)
    fresh = [s for s in samples if s.task_id not in excluded]
    fresh.sort(key=lambda s: _task_id_sort_key(s.task_id))
    if limit and limit > 0:
        fresh = fresh[:limit]
    return fresh


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-generation scheduling benchmark")
    parser.add_argument("--datasets", nargs="+", default=["humaneval", "math"],
                        help="数据集列表 (humaneval, mbpp, gsm8k, math 及新鲜题键 gsm8k_train, mbpp_train)")
    parser.add_argument("--limit", type=int, default=100,
                        help="每个数据集最大题数（-1=全量）；历史题排除后再截取")
    parser.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS),
                        choices=list(ALL_CONDITIONS), help="要运行的条件")
    parser.add_argument("--max-steps", type=int, default=2, help="Agent 最大协作轮数")
    parser.add_argument("--workers", type=int, default=1, help="并发 worker 数")
    parser.add_argument("--online", action="store_true", help="允许联网拉取 HF 数据集缓存")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="独立产物目录，例如 results/pre_generation_scheduling_20261117")
    parser.add_argument("--exclude-manifest", nargs="+", default=[],
                        help="历史结果 JSON；其中所有 task_id 会被排除")
    parser.add_argument("--exclude-task-id", nargs="+", default=[],
                        help="直接指定要排除的 task_id（如 HumanEval/32）")
    parser.add_argument("--allow-unexcluded-history", action="store_true",
                        help="允许在未排除历史题的情况下运行（仅用于本地冒烟）")
    parser.add_argument("--random-seed", type=int, default=20260812,
                        help="random_skip 与 bootstrap 的随机种子")
    parser.add_argument("--model", default="deepseek-chat", help="逻辑模型名称")
    parser.add_argument("--provider", default=None, choices=["deepseek", "private"],
                        help="强制指定后端提供方")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="统一温度（协议固定为 0）")
    parser.add_argument("--max-tokens", type=int, default=None, help="统一输出上限")
    parser.add_argument("--router-beta0", type=float, default=None)
    parser.add_argument("--router-lambda", type=float, default=None, dest="router_lambda")
    parser.add_argument("--router-gamma", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    ledger_dir = output_dir / "requests"
    if output_dir.exists() or ledger_dir.exists():
        print(f"Error: 输出目录已存在，拒绝覆盖: {output_dir}")
        return 1

    excluded = load_exclusions(args.exclude_manifest, args.exclude_task_id)
    if not args.allow_unexcluded_history:
        missing = [
            ds for ds in args.datasets
            if ds.lower() in {d.lower() for d in HISTORY_SENSITIVE_DATASETS} and not excluded
        ]
        if missing:
            print(
                "Error: 未提供历史题排除清单，拒绝在 "
                f"{', '.join(missing)} 上运行（否则无法主张“新鲜题”）。"
                "请用 --exclude-manifest/--exclude-task-id，或用 --allow-unexcluded-history 冒烟。"
            )
            return 1

    offline = not args.online
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["HF_DATASETS_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
    print(f"[HF Mode] {'OFFLINE' if offline else 'ONLINE'}")

    try:
        probe = build_client(args.model, args.provider, temperature=args.temperature,
                             max_tokens=args.max_tokens, log=True)
        llm_model, llm_provider, api_key = probe.model, probe.provider, probe.api_key
        del probe
    except (ValueError, KeyError) as exc:
        print(f"Error: LLM provider 初始化失败：{exc}")
        return 1

    inference_protocol = InferenceProtocol(temperature=args.temperature, max_tokens=args.max_tokens)
    experiment_config = ExperimentConfig()
    harness = MultiDatasetHarness(
        api_key=api_key,
        max_steps=args.max_steps,
        workers=args.workers,
        experiment_config=experiment_config,
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_temperature=args.temperature,
        llm_max_tokens=args.max_tokens,
        inference_protocol=inference_protocol,
        request_ledger_dir=str(ledger_dir),
        router_beta0=args.router_beta0,
        router_lambda=args.router_lambda,
        router_gamma=args.router_gamma,
    )

    VIBSparseCommRouter.warmup_shared_encoder()

    records: List[Dict[str, Any]] = []
    global_start = time.time()
    for ds in args.datasets:
        samples = _select_samples(ds, args.limit, excluded)
        print(f"\n{'=' * 60}\n Running {ds.upper()}: {len(samples)} 题（已排除历史题）\n{'=' * 60}")
        if not samples:
            print(f"  [Skipped] {ds} 无可运行题（全被排除或数据缺失）")
            continue
        for idx, sample in enumerate(samples, start=1):
            task_start = time.time()
            record = run_task(
                harness, sample, args.conditions, args.random_seed,
            )
            record["execution_time"] = time.time() - task_start
            records.append(record)
            flags = " ".join(
                f"{c}={'P' if record['conditions'][c]['passed'] else 'F'}"
                for c in args.conditions if c in record["conditions"]
            )
            print(f"  [{idx}/{len(samples)}] {sample.task_id}: {flags}")

    summary = summarize_conditions(records, args.conditions)
    comparisons = {
        "coverage_minus_full": compare_conditions(records, "coverage", "full", summary, seed=args.random_seed),
        "coverage_minus_post_gate": compare_conditions(records, "coverage", "post_gate", summary, seed=args.random_seed),
        "coverage_minus_random_skip": compare_conditions(records, "coverage", "random_skip", summary, seed=args.random_seed),
        "coverage_minus_fixed_trim": compare_conditions(records, "coverage", "fixed_trim", summary, seed=args.random_seed),
    }

    report = {
        "protocol": "paper/pre_generation_scheduling_protocol_zh.md",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "datasets": args.datasets,
            "limit": args.limit,
            "conditions": list(args.conditions),
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "random_seed": args.random_seed,
            "model": llm_model,
            "provider": llm_provider,
            "excluded_task_count": len(excluded),
            "closer_roles": CLOSER_ROLES,
        },
        "inference_protocol": inference_protocol.describe(),
        "summary": summary,
        "comparisons": comparisons,
        "total_execution_time": time.time() - global_start,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}\n 结果摘要\n{'=' * 60}")
    for cond in args.conditions:
        s = summary[cond]
        print(f"  {cond:12s} acc={s['accuracy_pct']:6.2f}%  msgs={s['messages']:6d}  "
              f"prompt={s['prompt_tokens']:8d}  total={s['total_tokens']:8d}  err={s['errors']}")
    for name, cmp in comparisons.items():
        acc = cmp["accuracy_delta_pp"]
        if acc:
            print(f"  {name}: Δacc={acc['mean']:+.2f}pp [{acc['lo']:+.2f},{acc['hi']:+.2f}] "
                  f"Δtotal={cmp['reduction_pct']['total_tokens']}")
    print(f"\n📄 产物已保存至: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

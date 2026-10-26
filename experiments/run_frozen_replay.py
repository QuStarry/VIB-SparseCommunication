"""在共同候选流上做有预算上限、可续跑的探索性接收端干预。

支持 `--dataset math|humaneval`：数学题按整数答案严格匹配，代码题按题面
测试用例实际执行评分。两者都只复用历史候选，只改变合成器接收到的消息。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.llm_client import build_client
from src.vib_router import VIBSparseCommRouter

POLICIES = ("full", "random", "cosine", "vib", "first_round", "direct")
BOUNDARY_POLICIES = ("full", "first_round", "first_round_control", "direct")
SCALAR = re.compile(r"[-+]?\d+(?:\.\d+)?")
# 复用既有历史轨迹，因此只覆盖两轮六候选齐全的任务。
DATASETS = {
    "math": {"roles": r"^\[(Planner|Solver|Verifier)\]: ", "max_tokens": 256},
    "humaneval": {"roles": r"^\[(Architect|Coder|Reviewer)\]: ", "max_tokens": 1024},
}


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def strict_scalar_equal(output, reference):
    """仅匹配完整标量，不从公式、列表或解释中拾取末位数字。"""
    answer = re.sub(r"^(?:答案|Answer)\s*[:：]\s*", "", output.strip(), flags=re.I)
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        answer = answer[7:-1].strip()
    if not SCALAR.fullmatch(answer) or not SCALAR.fullmatch(reference.strip()):
        return False
    try:
        return Decimal(answer) == Decimal(reference.strip())
    except InvalidOperation:
        return False


def select_tasks(traces, count, seed, dataset, excluded_ids=()):
    """按题号哈希抽样，不读取历史正确率；限完整的六候选轨迹。"""
    roles = _dataset_config(dataset)["roles"]
    eligible = [
        row for row in traces
        if row["task_id"] not in excluded_ids
        and len(row["full"]["history"]) == 6
        and all(re.match(roles, message) for message in row["full"]["history"])
        and _has_reference(row["full"], dataset)
    ]
    if count < 1 or count > len(eligible):
        raise ValueError("任务数量超出可用轨迹范围")
    if len({row["task_id"] for row in eligible}) != len(eligible):
        raise ValueError("重复 task_id")
    eligible.sort(key=lambda row: fingerprint([seed, row["task_id"]]))
    return eligible[:count]


def _dataset_config(name):
    if name not in DATASETS:
        raise ValueError(f"未知数据集 {name!r}；可选 {sorted(DATASETS)}")
    return DATASETS[name]


def _has_reference(full, dataset):
    """数学题需要整数参考答案；代码题需要可执行评测入口。"""
    if dataset == "math":
        return bool(re.fullmatch(r"[-+]?\d+", str(full["reference_answer"]).strip()))
    return bool(full.get("entry_point")) and bool(full.get("test_cases"))


def build_task(row, dataset):
    """只保留重放所需字段；评测材料留在评测侧，不进入任何提示词。"""
    full = row["full"]
    task = {
        "task_id": row["task_id"],
        "problem": full["prompt"],
        "candidates": full["history"],
    }
    if dataset == "math":
        task["reference"] = full["reference_answer"]
    else:
        task["entry_point"] = full["entry_point"]
        task["test_cases"] = full["test_cases"]
    return task


def build_prompt(dataset, task, history):
    if dataset == "math":
        return synthesis_prompt(task["problem"], history)
    from experiments.run_multi_benchmark import build_final_code_prompt
    return build_final_code_prompt(task["problem"], history)


def score_output(dataset, task, output, evaluation_seed=None):
    if dataset == "math":
        return strict_scalar_equal(output, task["reference"])
    from experiments.run_multi_benchmark import CodeEvaluator, sanitize_code_output
    code = sanitize_code_output(output)
    if evaluation_seed is None:
        return CodeEvaluator.evaluate(code, task["test_cases"], entry_point=task["entry_point"])
    # Seed in the execution child, not the parent: no global RNG pollution.
    code = f"import random\nrandom.seed({evaluation_seed})\nexec({code!r}, globals())"
    tests = [f"import random\nrandom.seed({evaluation_seed})", *task["test_cases"]]
    first = CodeEvaluator.evaluate(code, tests, entry_point=task["entry_point"])
    second = CodeEvaluator.evaluate(code, tests, entry_point=task["entry_point"])
    if first != second:
        raise ValueError("固定评分种子下独立执行结果不一致")
    return first


def admitted_history(messages, goal, policy, task_id, seed):
    """冻结同轮历史；这里只干预接收端，不能等同生成前调度。"""
    if policy == "full":
        return list(messages), []
    if policy in ("first_round", "first_round_control"):
        return list(messages[:3]), []
    if policy == "direct":
        return [], []
    router = VIBSparseCommRouter(
        beta_0=3.2, lambda_decay=0.08, temp_tau=0.5, gamma=-0.4,
        adaptive=True, alpha=0.35, tau_min=-1.0,
        gating_mode={"cosine": "topk"}.get(policy, policy),
        keep_rate=0.70, sim_threshold=0.75,
        enable_math_role_weights=False,
        enable_reasoning_guard=False,
        enable_math_critical_guard=True,
        random_seed=VIBSparseCommRouter.derive_task_seed(seed, task_id),
    )
    history, decisions = [], []
    for offset in range(0, len(messages), 3):
        frozen = list(history)
        approved = []
        for index in range(offset, offset + 3):
            message = messages[index]
            match = re.match(r"^\[([^]]+)\]: (.*)$", message, re.S)
            result = router.evaluate_detailed(
                candidate_msg=match[2], history=frozen, goal=goal,
                step_t=offset // 3 + 1, role=match[1],
            )
            decisions.append({"candidate_index": index, **asdict(result)})
            if result.is_allowed:
                approved.append(message)
        history.extend(approved)
    return history, decisions


def synthesis_prompt(problem, history):
    return (
        "请根据团队讨论，给出以下数学问题的最终答案。\n\n"
        f"【问题】:\n{problem}\n\n【团队讨论历史】:\n"
        + ("\n".join(history) if history else "(无历史讨论)")
        + "\n\n【输出要求】:\n1. 直接给出最终数字答案，不要多余解释。\n"
        "2. 格式：答案: <数字>"
    )


def proxy_cost(prompt_tokens, completion_tokens, input_rate, output_rate):
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1e6


def aggregate(records, policies=POLICIES):
    """重复合成先按任务聚合；不把调用次数当独立样本数。"""
    summary = {}
    for policy in policies:
        rows = [row for row in records if row["policy"] == policy]
        valid = [row for row in rows if row["status"] == "ok"]
        tasks = {}
        for row in valid:
            tasks.setdefault(row["task_id"], []).append(int(row["passed"]))
        summary[policy] = {
            "requests": len(rows), "valid_requests": len(valid),
            "tasks": len(tasks),
            "task_mean_accuracy": (
                sum(sum(values) / len(values) for values in tasks.values()) / len(tasks)
                if tasks else None
            ),
            "prompt_tokens": sum(row.get("prompt_tokens", 0) for row in rows),
            "completion_tokens": sum(row.get("completion_tokens", 0) for row in rows),
            "proxy_usd": sum(row["charged_proxy_usd"] for row in rows),
            "mean_admitted_messages": (
                sum(row["admitted_messages"] for row in rows) / len(rows) if rows else None
            ),
            "mean_elapsed_ms": (
                sum(row.get("elapsed_ms", 0) for row in rows) / len(rows) if rows else None
            ),
        }
    return summary


def run(args):
    source = json.loads(args.source.read_text())
    config = _dataset_config(args.dataset)
    protocol = getattr(args, "protocol", "gating")
    policies = BOUNDARY_POLICIES if protocol == "boundary" else POLICIES
    excluded_ids = set()
    exclusions = []
    for path in getattr(args, "exclude_manifest", ()):
        prior = json.loads(path.read_text())
        if prior.get("dataset", "math") != args.dataset:
            raise ValueError("排除 manifest 的数据集不匹配")
        ids = [task["task_id"] for task in prior["tasks"]]
        excluded_ids.update(ids)
        exclusions.append({
            "source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_ids": ids,
        })
    if protocol == "boundary" and not exclusions:
        raise ValueError("边界重放必须明确排除旧 pilot manifest")
    selected = select_tasks(
        source["trace_results"][args.dataset], args.tasks, args.seed, args.dataset,
        excluded_ids,
    )
    tasks = [build_task(row, args.dataset) for row in selected]
    # 参考答案/测试用例仅存于评测侧；接收历史和合成提示词均不读取它。
    contract = {
        "version": 3 if protocol == "boundary" else 2,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "tasks": tasks, "seed": args.seed, "repeats": args.repeats,
        "model": args.model, "provider": args.provider, "temperature": 0.1,
        "max_tokens": config["max_tokens"], "policies": policies,
        "dataset": args.dataset,
        # 主 harness 的代码合成还会跑一轮只看题面示例的 doctest 自修复；它不读讨论历史，
        # 这里省去该回路以便对照对历史变化更敏感，属于有意的偏离。
        "synthesis_mode": "single_shot",
        "input_usd_per_million": args.input_rate,
        "output_usd_per_million": args.output_rate,
        "max_proxy_usd": args.max_usd,
        "scope": f"exploratory_{args.dataset}_subset_receiver_only_not_held_out",
        "router_config": {
            "beta_0": 3.2, "lambda_decay": 0.08, "temp_tau": 0.5,
            "gamma": -0.4, "adaptive": True, "alpha": 0.35, "tau_min": -1.0,
            "enable_math_role_weights": False, "enable_reasoning_guard": False,
            "enable_math_critical_guard": True, "same_round_history": "frozen",
            "random_keep_rate": 0.70, "cosine_threshold": 0.75,
        },
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "router_sha256": hashlib.sha256((ROOT / "src/vib_router.py").read_bytes()).hexdigest(),
    }
    if protocol == "boundary":
        contract.update(
            protocol="boundary", exclusions=exclusions,
            primary_contrast="full_minus_first_round",
            noise_control="first_round_control_minus_first_round",
            evaluation={
                "python_random_seed": args.seed,
                "independent_code_checks": 2,
                "numpy_and_other_rngs_controlled": False,
            },
            inference_claim="exploratory_new_replay_subset_not_held_out_no_equivalence_margin",
        )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(contract)):
        raise ValueError("续跑协议不一致，请使用新的输出目录")
    write_json(manifest, contract)
    records = [json.loads(path.read_text()) for path in args.output.glob("request_*.json")]
    if any(row["status"] != "ok" for row in records):
        raise RuntimeError("存在失败或中断请求，需人工审计；不自动重复计费")
    used = sum(row["charged_proxy_usd"] for row in records)
    if args.dry_run:
        print(json.dumps({"tasks": len(tasks), "max_requests": len(tasks)*len(policies)*args.repeats,
                          "max_proxy_usd": args.max_usd, "contract_sha256": fingerprint(contract)}))
        return
    client = build_client(args.model, args.provider)
    api = client.client.with_options(max_retries=0, timeout=60.0)
    jobs = [(task, repeat, policy) for task in tasks
            for repeat in range(args.repeats) for policy in policies]
    random.Random(args.seed).shuffle(jobs)
    stop_reason = "complete"
    for task, repeat, policy in jobs:
        request_id = fingerprint([fingerprint(contract), task["task_id"], repeat, policy])
        path = args.output / f"request_{request_id}.json"
        if path.exists():
            continue
        history, decisions = admitted_history(
            task["candidates"], task["problem"], policy, task["task_id"], args.seed
        )
        prompt = build_prompt(args.dataset, task, history)
        # UTF-8 字节数加余量作保守代理预留，失败请求也不假定免费。
        reserve = proxy_cost(
            len(prompt.encode()) + 1024, config["max_tokens"],
            args.input_rate, args.output_rate,
        )
        if used + reserve > args.max_usd:
            stop_reason = "budget_limit"
            break
        record = {
            "request_id": request_id, "task_id": task["task_id"], "repeat": repeat,
            "policy": policy, "status": "pending", "charged_proxy_usd": reserve,
            "admitted_messages": len(history), "decisions": decisions,
            "prompt_sha256": fingerprint(prompt),
        }
        write_json(path, record)
        started = time.monotonic()
        try:
            response = api.chat.completions.create(
                model=client.model, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=config["max_tokens"],
            )
            if response.usage:
                record.update(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    usage=response.usage.model_dump(),
                )
                record["charged_proxy_usd"] = proxy_cost(
                    record["prompt_tokens"], record["completion_tokens"],
                    args.input_rate, args.output_rate,
                )
            record["response_model"] = response.model
            if protocol == "boundary" and response.model != args.model:
                raise ValueError("返回模型与冻结 pinned 模型不一致")
            record["response_id"] = response.id
            record["finish_reason"] = response.choices[0].finish_reason
            record["output"] = response.choices[0].message.content
            if (not response.usage or not record["output"]
                    or record["finish_reason"] != "stop"):
                raise ValueError("用量缺失、空内容或输出截断；中止而不计作错误答案")
            evaluation_seed = args.seed if protocol == "boundary" else None
            record["passed"] = score_output(
                args.dataset, task, record["output"], evaluation_seed=evaluation_seed
            )
            if protocol == "boundary":
                record["evaluation_seed"] = evaluation_seed
            record["status"] = "ok"
        except Exception as error:
            record["status"] = "error"
            record["error_type"] = type(error).__name__
            stop_reason = "request_failure"
        record["elapsed_ms"] = 1000 * (time.monotonic() - started)
        write_json(path, record)
        records.append(record)
        used += record["charged_proxy_usd"]
        print(json.dumps({key: record[key] for key in (
            "task_id", "policy", "status", "charged_proxy_usd", "elapsed_ms"
        )}), flush=True)
        write_json(args.output / "summary.json", {
            "contract_sha256": fingerprint(contract), "stop_reason": stop_reason,
            "proxy_usd": used, "policies": aggregate(records, policies),
            "cost_scope": "synthesis_only_no_candidate_generation_not_actual_invoice",
        })
        if record["status"] != "ok":
            raise RuntimeError("请求失败，已持久化账目并停止")
    write_json(args.output / "summary.json", {
        "contract_sha256": fingerprint(contract), "stop_reason": stop_reason,
        "proxy_usd": used, "policies": aggregate(records, policies),
        "cost_scope": "synthesis_only_no_candidate_generation_not_actual_invoice",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/main_with_math_merged.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="math")
    parser.add_argument("--protocol", choices=("gating", "boundary"), default="gating")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--model", default="deepseek-v4-flash-0817")
    parser.add_argument("--provider", default="private")
    parser.add_argument("--input-rate", type=float, default=0.30)
    parser.add_argument("--output-rate", type=float, default=1.20)
    parser.add_argument("--max-usd", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.input_rate, args.output_rate, args.max_usd) <= 0:
        parser.error("重复数和费用参数必须为正")
    run(args)


if __name__ == "__main__":
    main()

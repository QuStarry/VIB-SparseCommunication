"""Separate receiver filtering quality, communication counts and actual usage."""

import argparse
import json
import os
from pathlib import Path

from experiments.analyze_prompt_contrast_replication import replication_statistics
from experiments.calibrate_prompt_protocol import _cost
from experiments.run_gate_receiver import validate_gate_session
from experiments.run_prompt_contrast_pilot import response_path, write_json
from src.experiment_driver import file_sha256
from src.exploratory_prompt_session import validate_exploratory_generation
from src.isolated_code_evaluation import load_evaluators
from src.request_ledger import fingerprint


NAMES = {"full": "Full全部接收", "vib": "VIB实际门控",
         "random_matched": "随机消息数匹配", "cosine_matched": "余弦消息数匹配"}


def communication_summary(manifest):
    unique = {(job["task_id"], job["prompt_variant"]): job for job in manifest["jobs"]}
    summary = {}
    for policy in manifest["policies"]:
        jobs = [job for (task, arm), job in unique.items() if arm == policy]
        matched = all(job["round_counts"] == unique[(job["task_id"], "vib")]["round_counts"]
                      for job in jobs)
        summary[policy] = {
            "messages": sum(len(job["admitted_indices"]) for job in jobs),
            "utf8_history_bytes": sum(job["admitted_history_bytes"] for job in jobs),
            "tasks": len(jobs), "matched_per_task_per_round": matched,
            "measurement_unit": "unique_task_history_not_repetition_weighted",
        }
    return summary


def _readme(output, result):
    primary = result["primary_comparison"]
    lower, upper = primary["ci95_pp"]
    summary = result["generation_summary"]
    text = [
        "# 共同候选门控接收端对照", "",
        "从前两轮72题之外固定抽32题，共同历史Full六候选，四种接收策略各重复两次，"
        "共256次预声明请求。全部使用同一冻结规格优先提示，温度0、输出cap1024，"
        "串行single-attempt；生成全部结束后才载入隐藏测试，完整输出两次独立解释器检查。",
        "", "## 真实结果", "",
        "| 接收策略 | 成功/64次 | 32题均值准确率 | 接收消息/192条 | 历史字节 | 输入Token | 输出Token | 历史代理费用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, group in result["statistics"]["groups"].items():
        cost, communication = result["costs"][policy], result["communication"][policy]
        text.append(f'| {NAMES[policy]} | {group["passed_generations"]}/64 | '
                    f'{group["task_mean_accuracy_pct"]:.3f}% | {communication["messages"]} | '
                    f'{communication["utf8_history_bytes"]} | {cost["prompt_tokens"]} | '
                    f'{cost["completion_tokens"]} | ${cost["proxy_usd"]} |')
    text.extend([
        "", "消息与字节计数按每题唯一历史，不把两次重复算双倍。"
        "每题两次评分先均值聚合，统计独立单位32题，不是best-of-two。",
        "", "## 预声明主比较", "",
        f'VIB减随机消息数匹配：{primary["delta_pp"]:+.3f}pp；'
        f'10000次任务级bootstrap95%区间[{lower:+.3f}, {upper:+.3f}]。',
        f'VIB更高{primary["left_higher_tasks"]}题，随机更高{primary["right_higher_tasks"]}题，'
        f'平局{primary["tied_tasks"]}题；双侧精确方向检验p={primary["exact_sign_p"]:.6f}。',
        "", "主比较预声明alpha0.05；均值bootstrap与任务方向检验不是同一统计量，"
        "不能只挑有利结果。其他比较均探索性未校正；跨零不表示等效，没有非劣margin。",
        "", "## 其他探索性比较", "",
        "| 左组减右组 | 差值pp | bootstrap95%区间 | 精确方向p |",
        "|---|---:|---|---:|",
    ])
    for comparison in result["statistics"]["comparisons"]:
        if {comparison["left"], comparison["right"]} == {"vib", "random_matched"}:
            continue
        lo, hi = comparison["ci95_pp"]
        text.append(f'| {NAMES[comparison["left"]]} - {NAMES[comparison["right"]]} | '
                    f'{comparison["delta_pp"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] | '
                    f'{comparison["exact_sign_p"]:.6f} |')
    text.extend([
        "", "## 编码与匹配协议", "",
        "真实本地`all-MiniLM-L6-v2`，snapshot "
        "`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，CPU/eval，384维，L2归一化。"
        "实际max_seq_length256，长消息按tokenizer截断；不是完整长代码语义表示。"
        "模型文件、库版本、向量与逐消息utility/阈值/历史均冻结在设计证据中。",
        "", "VIB按现有主代码incremental历史逐条评估，保留代码差异、纠错关键词、"
        "代码角色beta、复杂度与adaptive代理等已有启发式；不是单独的信息瓶颈公式消融。"
        "无新增摘要、首轮最小角色数、隐藏反馈或自修复。",
        "", "随机在每轮三条候选中不放回抽k，k精确等于该题该轮VIB通过数。"
        "余弦逐条贪心选择与已选历史最大相似度最小的k条，稳定按原消息顺序输出。"
        "这是数量匹配的条件对照；k来自冻结VIB决策，不是独立先验常数。"
        "不是原router的Bernoulli random或threshold topk配置，未改核心实现。",
        "", "## 预算与因果边界", "",
        "所有组每题两次调用、每次1024输出cap相同；随机和余弦逐题逐轮消息数量精确匹配VIB。"
        "消息长度、历史字节和输入Token不相等，因此不能称等实际费用对照。",
        f'本轮真实总计{summary["total_tokens"]} Token，历史代理金额${summary["proxy_cost_usd"]}；'
        f'截断{summary["truncated_responses"]}次，usage完整：{summary["usage_complete"]}。',
        "价格为2026-10-23输入$0.30/百万、输出$1.20/百万；$1为观察停机阈值，"
        "未取得实际美元/人民币invoice，不保证invoice硬上限。",
        "", "只改变已有候选的接收历史。第二轮候选仍是历史Full生成，"
        "没有重生成删消息后Agent本应看到的上下文，因此不是完整协作闭环的VIB/Full因果证明。"
        "候选生成开销已经发生，不在本轮新增账单中，不能声称端到端成本节省。",
        "", "来源仍是已审计HumanEval存档，不重叠于前两轮并不构成全新独立数据，"
        "不能排除预训练污染。模型ID匹配仅证明本轮返回声明，不证明官方immutable snapshot。",
        "", "## 工作流", "", "```mermaid", "flowchart LR",
        "A[固定32题共同六候选] --> B[真实编码冻结逐条门控]",
        "B --> C[VIB及逐轮数量匹配随机和余弦]", "C --> D[同提示256次合成并保存usage]",
        "D --> E[结束生成后隔离评分]", "E --> F[质量消息预算实际成本分开报告]", "```",
        "", "## 一键导航", "", "| 内容 | 入口 |", "|---|---|",
        "| 编码向量与门控决策 | [design.json](../gate_receiver_design_20261028/design.json) |",
        "| 离线设计摘要与hash | [design_receipt.json](../gate_receiver_design_20261028/design_receipt.json) |",
        "| 冻结请求与预算 | [manifest.json](manifest.json) |",
        "| 真实身份与usage | [generation_report.json](generation_report.json) |",
        "| pending及终态账目 | [requests.jsonl](requests.jsonl) |",
        "| 完整原文 | [responses/](responses/) |",
        "| 逐题评分及统计 | [analysis.json](analysis.json) |", "",
    ])
    path = output / "README.md"
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(text))
            stream.flush()
            os.fsync(stream.fileno())


def analyze_gate_session(output_dir):
    """Evaluate all frozen outputs only after generation completes."""
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    validate_gate_session(manifest)
    generation_path = output / "generation_report.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    records = validate_exploratory_generation(manifest, output, generation)
    if not generation["summary"]["run_complete"] or not generation["summary"]["usage_complete"]:
        raise ValueError("Incomplete receiver generation cannot produce accuracy comparison")
    identity = {"manifest_sha256": fingerprint(manifest),
                "generation_report_sha256": file_sha256(generation_path),
                "implementation_sha256": file_sha256(__file__)}
    path = output / "analysis.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in stored.items() if key != "content_sha256"}
        if (any(stored.get(key) != value for key, value in identity.items())
                or fingerprint(payload) != stored.get("content_sha256")):
            raise ValueError("Cached receiver analysis changed")
        _readme(output, stored)
        return stored
    evaluators = load_evaluators(manifest["evaluation"]["path"], manifest["evaluation"]["sha256"])
    by_operation = {record["operation_id"]: record for record in records}
    manifest_id = fingerprint(manifest)
    rows, costs = [], {}
    for job in manifest["jobs"]:
        record = by_operation[fingerprint([manifest_id, job["job_id"]])]
        raw = json.loads(response_path(output, job).read_text(encoding="utf-8"))
        passed = False if record["status"] == "truncated" else evaluators[job["task_id"]](raw["content"])
        if type(passed) is not bool:
            raise ValueError("Scoring result must be boolean")
        rows.append({key: job[key] for key in ("job_id", "task_id", "prompt_variant", "repetition")}
                    | {"passed": passed, "status": record["status"], "output_sha256": raw["output_sha256"]})
        cost = costs.setdefault(job["prompt_variant"], {"prompt_tokens": 0, "completion_tokens": 0})
        for field in ("prompt_tokens", "completion_tokens"):
            cost[field] += record["usage"][field]
    for cost in costs.values():
        cost["proxy_usd"] = str(_cost(cost["prompt_tokens"], cost["completion_tokens"]))
    settings = manifest["analysis"]
    stats, primary = replication_statistics(rows, settings)
    result = {**identity, "statistics": stats, "primary_comparison": primary,
              "evaluations": rows, "costs": costs, "communication": communication_summary(manifest),
              "generation_summary": generation["summary"], "inference_scope": manifest["scope"]}
    validate_exploratory_generation(manifest, output, generation)
    if identity["generation_report_sha256"] != file_sha256(generation_path):
        raise ValueError("Generation report changed during evaluation")
    result["content_sha256"] = fingerprint(result)
    write_json(path, result)
    _readme(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_gate_session(args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "primary_comparison", "communication", "costs", "statistics")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Multi-Dataset Evaluation Harness for VIB-Sparse Communication

统一入口支持 HumanEval / MBPP / GSM8K 三大数据集的全量 Benchmark 评估，
对比 Full Comm (对照组) 与 VIB Sparse (我们的方法) 的 Pass@1、消息压缩率与 Token 节省。

Usage:
    python3 experiments/run_multi_benchmark.py --datasets humaneval mbpp gsm8k --limit 10
    python3 experiments/run_multi_benchmark.py --datasets humaneval --limit 164  # 全量
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import re
import json
import time
import gzip
import math
import argparse
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import DeepSeekClient
from llm_agents import LLMAgent
from vib_router import VIBSparseCommRouter, VIBEvaluationResult
from workflow import VIBMultiAgentWorkflow


def _is_hf_offline_mode() -> bool:
    return (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("HF_DATASETS_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )

#region debug-point dbg_helpers
_DBG_RUN_ID = str(int(time.time() * 1000))

def _dbg_get_url() -> Optional[str]:
    url = os.environ.get("DEBUG_SERVER_URL")
    if url:
        return url

    env_path = os.path.join(os.path.dirname(__file__), "..", ".dbg", "mbpp-eval-fails.env")
    if not os.path.exists(env_path):
        return None

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("DEBUG_SERVER_URL="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        return None
    return None

def _dbg_get_session_id() -> Optional[str]:
    sid = os.environ.get("DEBUG_SESSION_ID")
    if sid:
        return sid

    env_path = os.path.join(os.path.dirname(__file__), "..", ".dbg", "mbpp-eval-fails.env")
    if not os.path.exists(env_path):
        return None

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("DEBUG_SESSION_ID="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        return None
    return None

def _dbg_event(event_type: str, payload: Dict[str, Any]):
    url = _dbg_get_url()
    if not url:
        return
    sid = _dbg_get_session_id()

    try:
        import urllib.request
        data = json.dumps({
            "sessionId": sid,
            "eventType": event_type,
            "runId": _DBG_RUN_ID,
            "ts": time.time(),
            "payload": payload
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1.0) as _:
            pass
    except Exception:
        return
#endregion debug-point dbg_helpers


# ============================================
# 1. 统一数据结构 (Data Structures)
# ============================================

@dataclass
class TaskSample:
    """标准化任务样本"""
    task_id: str
    dataset_name: str
    prompt: str
    entry_point: Optional[str] = None
    test_cases: Optional[List[str]] = None
    reference_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """单个 Task 的评估结果"""
    task_id: str
    dataset_name: str
    full_passed: bool
    vib_passed: bool
    full_messages: int
    vib_messages: int
    full_tokens: int
    vib_tokens: int
    execution_time: float
    error_msg: Optional[str] = None


# ============================================
# 2. 数据集加载适配器 (Dataset Adapters)
# ============================================

class BaseDatasetAdapter(ABC):
    @abstractmethod
    def load(self, limit: int = -1) -> List[TaskSample]:
        pass


class HumanEvalAdapter(BaseDatasetAdapter):
    """加载 HumanEval 164 题 (本地缓存或在线下载)"""
    DATASET_PATH = os.path.join(os.path.dirname(__file__), "HumanEval.jsonl.gz")
    HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

    def load(self, limit: int = -1) -> List[TaskSample]:
        if not os.path.exists(self.DATASET_PATH):
            if _is_hf_offline_mode():
                print(f"  [Warning] 离线模式下未找到 HumanEval 数据集: {self.DATASET_PATH}")
                print("  请先手动放置 HumanEval.jsonl.gz，或用 --online 运行一次以允许下载")
                return []
            print(f"  📥 下载 HumanEval 数据集...")
            import urllib.request
            urllib.request.urlretrieve(self.HUMANEVAL_URL, self.DATASET_PATH)

        samples = []
        with gzip.open(self.DATASET_PATH, 'rt', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                samples.append(TaskSample(
                    task_id=item["task_id"],
                    dataset_name="HumanEval",
                    prompt=item["prompt"],
                    entry_point=item["entry_point"],
                    test_cases=[item["test"]],
                    reference_answer=item["canonical_solution"]
                ))
                if 0 < limit <= len(samples):
                    break
        return samples


class MBPPAdapter(BaseDatasetAdapter):
    """加载 MBPP Sanitized 版本 (通过 HuggingFace datasets)"""

    def load(self, limit: int = -1) -> List[TaskSample]:
        try:
            from datasets import load_dataset
            ds = load_dataset("mbpp", "sanitized", split="test")
        except ImportError:
            print("  [Warning] 无法加载 MBPP 数据集: 缺少 datasets 依赖")
            print("  请安装依赖: pip install -r requirements.txt")
            return []
        except Exception as e:
            print(f"  [Warning] 无法加载 MBPP 数据集: {e}")
            if "OfflineModeIsEnabled" in str(e) or _is_hf_offline_mode():
                print("  当前为离线模式且本地未命中缓存；请先用 --online 运行一次以下载缓存")
            else:
                print("  请检查网络/代理，或确认 HuggingFace Hub 可访问")
            return []

        samples = []
        for item in ds:
            # 提取 entry_point 函数名
            code = item.get("code", "")
            entry_point = None
            if "def " in code:
                entry_point = code.split("def ")[1].split("(")[0].strip()

            signature_line = ""
            m = re.search(r"def\\s+(\\w+)\\s*\\(([^)]*)\\)\\s*:", code)
            if m:
                signature_line = f"def {m.group(1)}({m.group(2)}):"

            test_imports = item.get("test_imports", []) or []
            test_list = item.get("test_list", []) or []
            test_cases = [*test_imports, *test_list]

            prompt_text = "\n".join([
                "请根据题目描述实现函数，并确保函数名与签名严格匹配，能通过单元测试。",
                "",
                "【函数签名】",
                signature_line or (f"def {entry_point}(...):" if entry_point else "def <function>(...):"),
                "",
                "【题目】",
                item.get("prompt", ""),
                "",
                "【单元测试（供参考，不要改动测试的函数名）】",
                "\n".join(test_list[:5])
            ]).strip()

            samples.append(TaskSample(
                task_id=f"MBPP/{item['task_id']}",
                dataset_name="MBPP",
                prompt=prompt_text,
                entry_point=entry_point,
                test_cases=test_cases,
                reference_answer=code
            ))

            _dbg_event("dataset_sample", {
                "dataset": "MBPP",
                "task_id": f"MBPP/{item.get('task_id')}",
                "prompt_head": prompt_text[:200],
                "entry_point": entry_point,
                "test_imports": test_imports,
                "test_case_head": (test_list or [""])[0][:200],
                "signature_line": signature_line
            })

            if 0 < limit <= len(samples):
                break
        return samples


class GSM8KAdapter(BaseDatasetAdapter):
    """加载 GSM8K Main 测试集 (数学推理)"""

    def load(self, limit: int = -1) -> List[TaskSample]:
        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split="test")
        except ImportError:
            print("  [Warning] 无法加载 GSM8K 数据集: 缺少 datasets 依赖")
            print("  请安装依赖: pip install -r requirements.txt")
            return []
        except Exception as e:
            print(f"  [Warning] 无法加载 GSM8K 数据集: {e}")
            if "OfflineModeIsEnabled" in str(e) or _is_hf_offline_mode():
                print("  当前为离线模式且本地未命中缓存；请先用 --online 运行一次以下载缓存")
            else:
                print("  请检查网络/代理，或确认 HuggingFace Hub 可访问")
            return []

        samples = []
        for i, item in enumerate(ds):
            # 解析 GSM8K 标答中 #### 后面的数字
            answer = item["answer"]
            target = answer.split("####")[-1].strip() if "####" in answer else answer.strip()

            samples.append(TaskSample(
                task_id=f"GSM8K/{i}",
                dataset_name="GSM8K",
                prompt=item["question"],
                reference_answer=target,
                metadata={"full_answer": answer}
            ))
            if 0 < limit <= len(samples):
                break
        return samples


class DatasetFactory:
    _adapters = {
        "humaneval": HumanEvalAdapter(),
        "mbpp": MBPPAdapter(),
        "gsm8k": GSM8KAdapter()
    }

    @classmethod
    def get_adapter(cls, name: str) -> BaseDatasetAdapter:
        key = name.lower()
        if key not in cls._adapters:
            raise ValueError(f"不支持的数据集: {name}。可选: {list(cls._adapters.keys())}")
        return cls._adapters[key]


# ============================================
# 3. 评估引擎 (Evaluators)
# ============================================

class CodeEvaluator:
    """Python 代码执行评估 (HumanEval & MBPP)"""
    @staticmethod
    def _unsafe_execute(code: str, test_script: str, entry_point: str, result_queue, dbg_meta: Optional[Dict[str, Any]] = None):
        try:
            if entry_point:
                full_code = f"{code}\n\n{test_script}\n\ncheck({entry_point})"
            else:
                full_code = f"{code}\n\n{test_script}"
            exec_globals = {}
            exec(full_code, exec_globals)
            result_queue.put(True)
        except Exception:
            #region debug-point code_exec_error
            try:
                import traceback
                _dbg_event("code_exec_error", {
                    "meta": dbg_meta or {},
                    "entry_point": entry_point,
                    "code_head": (code or "")[:400],
                    "code_tail": (code or "")[-400:],
                    "test_head": (test_script or "")[:400],
                    "traceback": traceback.format_exc(limit=20)
                })
            except Exception:
                pass
            #endregion debug-point code_exec_error
            result_queue.put(False)

    @classmethod
    def evaluate(cls, code_str: str, test_cases: List[str], entry_point: str = None,
                 timeout: float = 5.0, dbg_meta: Optional[Dict[str, Any]] = None) -> bool:
        test_script = "\n".join(test_cases)
        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=cls._unsafe_execute,
            args=(code_str, test_script, entry_point, result_queue, dbg_meta)
        )
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return False
        if not result_queue.empty():
            return result_queue.get()
        return False


class MathEvaluator:
    """GSM8K 数值匹配评估"""
    @staticmethod
    def evaluate(prediction: str, reference: str) -> bool:
        # 提取预测中最后一个数字
        extracted = re.findall(r"[-+]?\d*\.\d+|\d+", prediction)
        if not extracted:
            return False
        last_pred = extracted[-1].strip().rstrip(".")
        ref = reference.strip().replace(",", "")
        return last_pred == ref


class UnifiedEvaluator:
    @classmethod
    def evaluate(cls, sample: TaskSample, model_output: str) -> bool:
        if sample.dataset_name == "HumanEval":
            if not sample.test_cases:
                return False
            return CodeEvaluator.evaluate(
                model_output, sample.test_cases,
                entry_point=sample.entry_point,
                dbg_meta={"task_id": sample.task_id, "dataset": sample.dataset_name}
            )
        if sample.dataset_name == "MBPP":
            if not sample.test_cases:
                return False
            return CodeEvaluator.evaluate(
                model_output, sample.test_cases,
                entry_point=None,
                dbg_meta={"task_id": sample.task_id, "dataset": sample.dataset_name}
            )
        elif sample.dataset_name == "GSM8K":
            if not sample.reference_answer:
                return False
            return MathEvaluator.evaluate(model_output, sample.reference_answer)
        else:
            raise NotImplementedError(f"无评估器: {sample.dataset_name}")


# ============================================
# 4. Agent 系统构建 (复用 VIB 基础设施)
# ============================================

class BypassVIBRouter:
    """Full Comm 对照组：无条件放行所有消息"""
    def evaluate_detailed(self, candidate_msg: str, history: list, goal: str, step_t: int,
                          token_probs=None, confidence_score=None, role=None,
                          is_execution_failed=False) -> VIBEvaluationResult:
        return VIBEvaluationResult(
            utility_score=1.0, sufficiency=1.0, redundancy=0.0,
            beta_t=1.0, is_allowed=True, threshold=0.0
        )


def create_code_agents(client: DeepSeekClient) -> list:
    """创建代码类任务的 3-Agent 团队"""
    return [
        LLMAgent("Architect", "你是软件架构师。分析需求、边界条件与最佳算法策略。", client),
        LLMAgent("Coder", "你是核心程序员。根据需求编写高效 Python 代码。", client),
        LLMAgent("Reviewer", "你是代码审查员。严格检查边界漏洞与逻辑错误。", client)
    ]


def create_math_agents(client: DeepSeekClient) -> list:
    """创建数学推理任务的 3-Agent 团队"""
    return [
        LLMAgent("Planner", "你是数学规划师。分析题目结构，制定解题策略。", client),
        LLMAgent("Solver", "你是核心解题者。执行详细的数学计算与推导。", client),
        LLMAgent("Verifier", "你是验证专家。检查计算步骤的正确性与逻辑一致性。", client)
    ]


def generate_final_code(client: DeepSeekClient, prompt: str, history: list) -> str:
    """根据讨论历史生成最终代码"""
    history_text = "\n".join(history) if history else "(无历史讨论)"
    prompt_text = f"""请阅读团队讨论历史，补全并实现以下 Python 函数。

【需求与签名】:
{prompt}

【团队讨论历史】:
{history_text}

【极其重要的输出要求】:
1. 仅输出最终可执行的 Python 代码实现。
2. 必须包含完整的函数定义并直接可被调用。
3. 不要包含任何 markdown 说明文字或解说，直接写代码。"""

    res = client.chat(
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=1024, temperature=0.1
    )
    code = res["content"]
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    return code


def generate_final_answer(client: DeepSeekClient, prompt: str, history: list) -> str:
    """根据讨论历史生成最终数学答案"""
    history_text = "\n".join(history) if history else "(无历史讨论)"
    prompt_text = f"""请根据团队讨论，给出以下数学问题的最终答案。

【问题】:
{prompt}

【团队讨论历史】:
{history_text}

【输出要求】:
1. 直接给出最终数字答案，不要多余解释。
2. 格式：答案: <数字>"""

    res = client.chat(
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=256, temperature=0.1
    )
    return res["content"]


# ============================================
# 5. 多数据集 Benchmark Harness
# ============================================

class MultiDatasetHarness:
    def __init__(self, api_key: str, max_steps: int = 2):
        self.api_key = api_key
        self.max_steps = max_steps

    def _run_single_task(self, sample: TaskSample) -> TaskResult:
        """对单个任务执行 Full Comm + VIB Sparse 对比评估"""
        task_start = time.time()

        try:
            # 选择 Agent 类型
            if sample.dataset_name in ("HumanEval", "MBPP"):
                create_agents_fn = create_code_agents
                generate_fn = generate_final_code
            else:
                create_agents_fn = create_math_agents
                generate_fn = generate_final_answer

            # ---- Full Comm (对照组) ----
            client_full = DeepSeekClient(api_key=self.api_key, max_tokens=256)
            agents_full = create_agents_fn(client_full)
            workflow_full = VIBMultiAgentWorkflow(
                agents=[a.to_agent_dict() for a in agents_full],
                vib_router=BypassVIBRouter(),
                max_steps=self.max_steps
            )
            res_full = workflow_full.run(goal=sample.prompt, verbose=False)
            output_full = generate_fn(client_full, sample.prompt, res_full["history"])
            full_passed = UnifiedEvaluator.evaluate(sample, output_full)
            full_tokens = client_full.get_token_stats()["total_tokens"]
            full_msgs = workflow_full.stats["total_approved"]

            # ---- VIB Sparse (我们的方法) ----
            client_vib = DeepSeekClient(api_key=self.api_key, max_tokens=256)
            agents_vib = create_agents_fn(client_vib)
            router_vib = VIBSparseCommRouter(
                beta_0=3.2, lambda_decay=0.08, temp_tau=0.5,
                gamma=-0.4, adaptive=True, alpha=0.35, tau_min=-1.0
            )
            workflow_vib = VIBMultiAgentWorkflow(
                agents=[a.to_agent_dict() for a in agents_vib],
                vib_router=router_vib,
                max_steps=self.max_steps
            )
            res_vib = workflow_vib.run(goal=sample.prompt, verbose=False)
            output_vib = generate_fn(client_vib, sample.prompt, res_vib["history"])
            vib_passed = UnifiedEvaluator.evaluate(sample, output_vib)
            vib_tokens = client_vib.get_token_stats()["total_tokens"]
            vib_msgs = workflow_vib.stats["total_approved"]

            exec_time = time.time() - task_start

            return TaskResult(
                task_id=sample.task_id,
                dataset_name=sample.dataset_name,
                full_passed=full_passed,
                vib_passed=vib_passed,
                full_messages=full_msgs,
                vib_messages=vib_msgs,
                full_tokens=full_tokens,
                vib_tokens=vib_tokens,
                execution_time=exec_time
            )

        except Exception as e:
            exec_time = time.time() - task_start
            return TaskResult(
                task_id=sample.task_id,
                dataset_name=sample.dataset_name,
                full_passed=False, vib_passed=False,
                full_messages=0, vib_messages=0,
                full_tokens=0, vib_tokens=0,
                execution_time=exec_time,
                error_msg=str(e)
            )

    def run_benchmark(self, datasets: List[str], limit: int = -1) -> Dict[str, Any]:
        """运行多数据集 Benchmark 并输出对比报告"""
        all_results = {}
        summary = {}
        global_start = time.time()

        for ds_name in datasets:
            print(f"\n{'='*60}")
            print(f" 🚀 Running Benchmark: {ds_name.upper()}")
            print(f"{'='*60}")

            adapter = DatasetFactory.get_adapter(ds_name)
            samples = adapter.load(limit=limit)

            if not samples:
                print(f"  [Skipped] 无数据加载: {ds_name}")
                continue

            total = len(samples)
            print(f"  已加载 {total} 个样本")

            results: List[TaskResult] = []
            stats = {
                "full_passed": 0, "vib_passed": 0,
                "full_tokens": 0, "vib_tokens": 0,
                "full_msgs": 0, "vib_msgs": 0
            }

            for idx, sample in enumerate(samples):
                res = self._run_single_task(sample)
                results.append(res)

                # 累计统计
                stats["full_passed"] += int(res.full_passed)
                stats["vib_passed"] += int(res.vib_passed)
                stats["full_tokens"] += res.full_tokens
                stats["vib_tokens"] += res.vib_tokens
                stats["full_msgs"] += res.full_messages
                stats["vib_msgs"] += res.vib_messages

                # 进度输出
                elapsed = time.time() - global_start
                avg_time = elapsed / (idx + 1)
                eta = avg_time * (total - idx - 1) / 60.0

                full_icon = "✅" if res.full_passed else "❌"
                vib_icon = "✅" if res.vib_passed else "❌"
                print(f"  [{idx+1:03d}/{total:03d}] {res.task_id:<25} | "
                      f"Full: {full_icon} | VIB: {vib_icon} (消息 {res.vib_messages}/6) | "
                      f"耗时: {res.execution_time:.1f}s | ETA: {eta:.1f}min")

            # 汇总该数据集指标
            full_rate = (stats["full_passed"] / total) * 100
            vib_rate = (stats["vib_passed"] / total) * 100
            msg_compression = ((stats["full_msgs"] - stats["vib_msgs"]) / max(1, stats["full_msgs"])) * 100
            token_saving = ((stats["full_tokens"] - stats["vib_tokens"]) / max(1, stats["full_tokens"])) * 100

            summary[ds_name] = {
                "total_tasks": total,
                "full_pass_rate": round(full_rate, 2),
                "vib_pass_rate": round(vib_rate, 2),
                "full_messages": stats["full_msgs"],
                "vib_messages": stats["vib_msgs"],
                "msg_compression_pct": round(msg_compression, 2),
                "full_tokens": stats["full_tokens"],
                "vib_tokens": stats["vib_tokens"],
                "token_saving_pct": round(token_saving, 2)
            }
            all_results[ds_name] = [asdict(r) for r in results]

        # 打印总报告
        self._print_summary_report(summary)

        total_time = (time.time() - global_start) / 60.0
        print(f"\n评测总耗时: {total_time:.2f} 分钟")

        return {"summary": summary, "detailed_results": all_results}

    def _print_summary_report(self, summary: Dict[str, Any]):
        """打印多数据集对比报告"""
        print(f"\n{'='*80}")
        print(" 📊 Multi-Dataset Benchmark Report (Full Comm vs VIB Sparse)")
        print(f"{'='*80}")
        print(f"{'Dataset':<12} | {'Full Pass@1':>12} | {'VIB Pass@1':>12} | "
              f"{'Msg Compress':>13} | {'Token Saving':>13}")
        print("-" * 75)

        for ds_name, metrics in summary.items():
            print(f"{ds_name:<12} | {metrics['full_pass_rate']:>10.2f}% | "
                  f"{metrics['vib_pass_rate']:>10.2f}% | "
                  f"{metrics['msg_compression_pct']:>11.2f}% | "
                  f"{metrics['token_saving_pct']:>11.2f}%")

        print(f"{'='*80}")


# ============================================
# 6. CLI Entry
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset VIB Benchmark")
    parser.add_argument("--datasets", nargs="+", default=["humaneval"],
                        help="数据集列表 (humaneval, mbpp, gsm8k)")
    parser.add_argument("--limit", type=int, default=-1,
                        help="每个数据集最大样本数 (-1=全量)")
    parser.add_argument("--max-steps", type=int, default=2,
                        help="Agent 最大协作轮数")
    parser.add_argument("--online", action="store_true",
                        help="允许联网拉取/更新 HuggingFace 数据集缓存（默认离线）")
    parser.add_argument("--output", type=str, default="results/multi_benchmark.json",
                        help="输出 JSON 文件路径")
    args = parser.parse_args()

    offline = not args.online
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["HF_DATASETS_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"

    mode_text = "OFFLINE" if offline else "ONLINE"
    print(f"[HF Mode] {mode_text} (HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']}, HF_DATASETS_OFFLINE={os.environ['HF_DATASETS_OFFLINE']})")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: 请设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)

    harness = MultiDatasetHarness(api_key=api_key, max_steps=args.max_steps)
    report = harness.run_benchmark(datasets=args.datasets, limit=args.limit)

    # 保存 JSON 报告
    if args.output:
        output_path = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ds_tag = "_".join(args.datasets)
        output_path = os.path.join(
            os.path.dirname(__file__), "results",
            f"multi_benchmark_{ds_tag}_{timestamp}.json"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 详细结果已保存至: {output_path}")


if __name__ == "__main__":
    main()

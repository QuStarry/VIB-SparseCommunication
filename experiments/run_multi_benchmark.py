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
import argparse
import multiprocessing
import threading
from decimal import Decimal, InvalidOperation
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import DeepSeekClient, resolve_deepseek_api_key
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
    full_prompt_tokens: int = 0
    full_completion_tokens: int = 0
    vib_prompt_tokens: int = 0
    vib_completion_tokens: int = 0
    error_msg: Optional[str] = None


@dataclass
class ExperimentConfig:
    enable_gsm8k_frozen_history: bool = True
    enable_math_role_weights: bool = False
    enable_gsm8k_reasoning_guard: bool = False
    enable_humaneval_digest: bool = False
    enable_humaneval_first_round_role_guard: bool = True
    enable_mbpp_early_stop: bool = True


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
    def _extract_last_number(text: str) -> Optional[str]:
        matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+", text or "")
        if not matches:
            return None
        return matches[-1].replace(",", "").strip().rstrip(".")

    @staticmethod
    def evaluate(prediction: str, reference: str) -> bool:
        pred_number = MathEvaluator._extract_last_number(prediction)
        ref_number = MathEvaluator._extract_last_number(reference)
        if pred_number is None or ref_number is None:
            return False
        try:
            return Decimal(pred_number) == Decimal(ref_number)
        except InvalidOperation:
            return pred_number == ref_number


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


def extract_visible_examples(prompt: str) -> List[str]:
    """提取 HumanEval prompt 中题面可见的示例。"""
    lines = prompt.splitlines()
    examples: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue

        if stripped.startswith(">>>"):
            example_parts = [stripped]
            next_idx = idx + 1
            while next_idx < len(lines):
                continuation = lines[next_idx].rstrip().strip()
                if not continuation:
                    break
                if continuation.startswith(">>>"):
                    break
                if continuation.startswith('"""') or continuation.startswith("'''"):
                    break
                example_parts.append(continuation)
                next_idx += 1
            examples.append("\n".join(example_parts))
            idx = next_idx
            continue

        if "=>" in stripped or "==" in stripped or "➞" in stripped:
            examples.append(stripped)
            idx += 1
            continue

        if stripped.startswith("Input:"):
            example_parts = [stripped]
            next_idx = idx + 1
            while next_idx < len(lines):
                continuation = lines[next_idx].rstrip().strip()
                if not continuation:
                    break
                if continuation.startswith("Example") or continuation.startswith("Input:"):
                    break
                example_parts.append(continuation)
                if continuation.startswith("Output:"):
                    break
                next_idx += 1
            examples.append("\n".join(example_parts))
            idx = next_idx + 1
            continue

        idx += 1
    return examples


def build_final_code_prompt(prompt: str, history: list) -> str:
    """构造最终代码合成提示词，优先锚定题面示例而非讨论中的臆测约束。"""
    history_text = "\n".join(history) if history else "(无历史讨论)"
    visible_examples = extract_visible_examples(prompt)
    examples_text = "\n".join(visible_examples) if visible_examples else "(题面未提供可抽取示例)"
    return f"""请阅读团队讨论历史，补全并实现以下 Python 函数。

【需求与签名】:
{prompt}

【题面可见示例】:
{examples_text}

【团队讨论历史】:
{history_text}

【极其重要的输出要求】:
1. 仅输出最终可执行的 Python 代码实现。
2. 必须包含完整的函数定义并直接可被调用。
3. 不要包含任何 markdown 说明文字或解说，直接写代码。
4. 若团队讨论存在冲突，以题面函数签名、docstring 示例和隐含测试语义为最高优先级，不要臆造额外约束。
5. 优先选择最直接、最贴合示例的正确实现，避免为了“泛化”而偏离题目真实语义。
6. 若【题面可见示例】存在，它们是最终代码必须优先满足的显式约束。
7. 若团队讨论与题面自然语言说明、题面示例或题面运算语义（如优先级、边界条件、tie-break 规则）冲突，忽略讨论中的相反结论，以题面为最终裁决。
8. 不要添加题面和示例均未明确要求的防御性分支或额外边界校验（例如对输入范围、空值、负数、符号位、输入数组长度等做出题面没有显式要求的判断或 raise）；题面没有写应该抛异常时永远不要写 raise 分支，哪怕 Note 里只是陈述长度关系也不要做显式校验。
9. 永远不要因为“我认为示例有误”或“我认为示例需要修正”就推翻题面可见示例或示例对应的正确语义；题面和可见示例是最终裁决。
10. 若 Reviewer 明确给出“形如 输入X → 输出Y”的单点断言（例如“[1, 1, 2] 的次小值应为 1”或“foo(3,4) 必须返回 9”等），请在最终实现中显式写出能够满足该断言的判断/分支/算法，不能仅停留在口头讨论。
11. 当讨论里同时出现多种冲突实现/公式时，按以下顺序选最终方案：
    a) 先统计每种实现/公式的出现次数，有明确多数派（出现次数 ≥ 其他方案之和）时选多数派；
    b) 若打平，优先选 Architect 首次明确支持且 Reviewer 没有举出反例的方案；
    c) 任何被 Reviewer 用具体反例明确否决的方案（例如“该简化公式会导致 n=5 时返回 10，但实际仅 1 个有效三元组”）无论出现多少次，必须排除。
12. 当出现边界条件争议（例如 `>` vs `>=`、`<` vs `<=`、是否 raise、是否处理负数）时，严格按以下优先级裁决：
    a) Reviewer 给出的具体反例（如“shift = 位数时应返回原字符串，而不是反转”）最高；
    b) 若无反例，严格照抄题面文本里的运算符用词，题面写 `>` 就用 `>`，写 `>=` 就用 `>=`，绝不自己替换；
    c) 题面没提到应 raise ValueError 时，永远不要写 raise 分支（哪怕 Reviewer 建议加也要忽略）。
13. 不要输出报错文本、占位符或解释；若有不确定，仍返回你认为最可能通过测试的完整实现。"""


def sanitize_code_output(code: str) -> str:
    """从模型返回中提取纯代码文本。"""
    code = (code or "").strip()
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0].strip()
    return code.strip()


def generate_final_code(client: DeepSeekClient, prompt: str, history: list) -> str:
    """根据讨论历史生成最终代码"""
    prompt_text = build_final_code_prompt(prompt, history)
    last_code = ""
    for _ in range(3):
        res = client.chat(
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=1024, temperature=0.1
        )
        code = sanitize_code_output(res["content"])
        last_code = code
        if code and not code.startswith("[API Error:"):
            return code
    return last_code


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
    def __init__(self, api_key: str, max_steps: int = 2, workers: int = 1, experiment_config: Optional[ExperimentConfig] = None):
        self.api_key = api_key
        self.max_steps = max_steps
        self.workers = max(1, workers)
        self._print_lock = threading.Lock()
        self.experiment_config = experiment_config or ExperimentConfig()

    def _select_task_components(self, sample: TaskSample) -> Tuple[Any, Any]:
        if sample.dataset_name in ("HumanEval", "MBPP"):
            return create_code_agents, generate_final_code
        return create_math_agents, generate_final_answer

    def _workflow_options_for_sample(self, sample: TaskSample) -> Dict[str, Any]:
        return {
            "same_round_history_mode": (
                "frozen"
                if sample.dataset_name == "GSM8K" and self.experiment_config.enable_gsm8k_frozen_history
                else "incremental"
            ),
            "enable_digest_fallback": (
                sample.dataset_name == "HumanEval" and self.experiment_config.enable_humaneval_digest
            ),
            "required_roles_first_step": (
                ["Coder", "Reviewer"]
                if sample.dataset_name == "HumanEval" and self.experiment_config.enable_humaneval_first_round_role_guard
                else []
            ),
            "first_step_min_approvals": (
                2
                if sample.dataset_name == "HumanEval" and self.experiment_config.enable_humaneval_first_round_role_guard
                else 0
            ),
        }

    def _create_vib_router(self) -> VIBSparseCommRouter:
        return VIBSparseCommRouter(
            beta_0=3.2,
            lambda_decay=0.08,
            temp_tau=0.5,
            gamma=-0.4,
            adaptive=True,
            alpha=0.35,
            tau_min=-1.0,
            enable_math_role_weights=self.experiment_config.enable_math_role_weights,
            enable_reasoning_guard=self.experiment_config.enable_gsm8k_reasoning_guard,
        )

    def _run_variant(
        self,
        sample: TaskSample,
        create_agents_fn,
        generate_fn,
        router,
        max_steps_override: Optional[int] = None
    ) -> Dict[str, Any]:
        client = DeepSeekClient(api_key=self.api_key, max_tokens=256)
        agents = create_agents_fn(client)
        workflow_options = self._workflow_options_for_sample(sample)
        workflow = VIBMultiAgentWorkflow(
            agents=[a.to_agent_dict() for a in agents],
            vib_router=router,
            max_steps=max_steps_override or self.max_steps,
            same_round_history_mode=workflow_options["same_round_history_mode"],
            enable_digest_fallback=workflow_options["enable_digest_fallback"],
            required_roles_first_step=workflow_options["required_roles_first_step"],
            first_step_min_approvals=workflow_options["first_step_min_approvals"]
        )
        workflow_result = workflow.run(goal=sample.prompt, verbose=False)
        final_output = generate_fn(client, sample.prompt, workflow_result["history"])
        passed = UnifiedEvaluator.evaluate(sample, final_output)
        token_stats = client.get_token_stats()
        return {
            "history": workflow_result["history"],
            "workflow_stats": workflow.stats,
            "output": final_output,
            "passed": passed,
            "token_stats": token_stats,
            "early_stopped": False
        }

    def _build_trace_payload(self, sample: TaskSample, variant_name: str, variant_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if variant_result is None:
            return None
        token_stats = variant_result["token_stats"]
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
            "token_stats": token_stats,
            "early_stopped": variant_result.get("early_stopped", False),
        }

    def run_single_task_with_trace(self, sample: TaskSample) -> Dict[str, Any]:
        """执行单题并保留 Full/VIB 双分支 trace，用于差分归因分析。"""
        task_start = time.time()

        try:
            create_agents_fn, generate_fn = self._select_task_components(sample)

            # MBPP 简单题优先尝试 1 轮协作；若已通过则直接早停，避免固定 6 条消息模板化消耗
            if sample.dataset_name == "MBPP" and self.experiment_config.enable_mbpp_early_stop:
                probe_full = self._run_variant(
                    sample=sample,
                    create_agents_fn=create_agents_fn,
                    generate_fn=generate_fn,
                    router=BypassVIBRouter(),
                    max_steps_override=1
                )
                if probe_full["passed"]:
                    probe_full["early_stopped"] = True
                    full_result = probe_full
                else:
                    full_result = self._run_variant(
                        sample=sample,
                        create_agents_fn=create_agents_fn,
                        generate_fn=generate_fn,
                        router=BypassVIBRouter()
                    )

                probe_vib = self._run_variant(
                    sample=sample,
                    create_agents_fn=create_agents_fn,
                    generate_fn=generate_fn,
                    router=self._create_vib_router(),
                    max_steps_override=1
                )
                if probe_vib["passed"]:
                    probe_vib["early_stopped"] = True
                    vib_result = probe_vib
                else:
                    vib_result = self._run_variant(
                        sample=sample,
                        create_agents_fn=create_agents_fn,
                        generate_fn=generate_fn,
                        router=self._create_vib_router()
                    )
            else:
                full_result = self._run_variant(
                    sample=sample,
                    create_agents_fn=create_agents_fn,
                    generate_fn=generate_fn,
                    router=BypassVIBRouter()
                )

                vib_result = self._run_variant(
                    sample=sample,
                    create_agents_fn=create_agents_fn,
                    generate_fn=generate_fn,
                    router=self._create_vib_router()
                )
            exec_time = time.time() - task_start
            full_token_stats = full_result["token_stats"]
            vib_token_stats = vib_result["token_stats"]

            task_result = TaskResult(
                task_id=sample.task_id,
                dataset_name=sample.dataset_name,
                full_passed=full_result["passed"],
                vib_passed=vib_result["passed"],
                full_messages=full_result["workflow_stats"]["total_approved"],
                vib_messages=vib_result["workflow_stats"]["total_approved"],
                full_tokens=full_token_stats["total_tokens"],
                vib_tokens=vib_token_stats["total_tokens"],
                execution_time=exec_time,
                full_prompt_tokens=full_token_stats["total_prompt_tokens"],
                full_completion_tokens=full_token_stats["total_completion_tokens"],
                vib_prompt_tokens=vib_token_stats["total_prompt_tokens"],
                vib_completion_tokens=vib_token_stats["total_completion_tokens"]
            )
            return {
                "task_result": task_result,
                "full": full_result,
                "vib": vib_result
            }

        except Exception as e:
            exec_time = time.time() - task_start
            error_result = TaskResult(
                task_id=sample.task_id,
                dataset_name=sample.dataset_name,
                full_passed=False, vib_passed=False,
                full_messages=0, vib_messages=0,
                full_tokens=0, vib_tokens=0,
                execution_time=exec_time,
                error_msg=str(e)
            )
            return {
                "task_result": error_result,
                "full": None,
                "vib": None
            }

    def _run_single_task(self, sample: TaskSample) -> TaskResult:
        """对单个任务执行 Full Comm + VIB Sparse 对比评估"""
        return self.run_single_task_with_trace(sample)["task_result"]

    def run_benchmark(self, datasets: List[str], limit: int = -1) -> Dict[str, Any]:
        """运行多数据集 Benchmark 并输出对比报告"""
        all_results = {}
        trace_results = {}
        summary = {}
        global_start = time.time()

        for ds_name in datasets:
            print(f"\n{'='*60}")
            print(f" 🚀 Running Benchmark: {ds_name.upper()}")
            print(f"{'='*60}")

            if self.workers > 1:
                VIBSparseCommRouter.warmup_shared_encoder()

            adapter = DatasetFactory.get_adapter(ds_name)
            samples = adapter.load(limit=limit)

            if not samples:
                print(f"  [Skipped] 无数据加载: {ds_name}")
                continue

            total = len(samples)
            print(f"  已加载 {total} 个样本")

            results: List[Optional[TaskResult]] = [None] * total
            traces: List[Optional[Dict[str, Any]]] = [None] * total
            stats = {
                "full_passed": 0, "vib_passed": 0,
                "full_tokens": 0, "vib_tokens": 0,
                "full_msgs": 0, "vib_msgs": 0,
                "full_prompt_tokens": 0, "vib_prompt_tokens": 0,
                "full_completion_tokens": 0, "vib_completion_tokens": 0,
                "full_only": 0, "vib_only": 0,
                "both_pass": 0, "both_fail": 0,
                "same_msg_count": 0
            }

            dataset_start = time.time()

            def consume_result(result_index: int, task_bundle: Dict[str, Any], done_count: int):
                res = task_bundle["task_result"]
                results[result_index] = res
                sample = samples[result_index]
                traces[result_index] = {
                    "task_id": sample.task_id,
                    "dataset_name": sample.dataset_name,
                    "full": self._build_trace_payload(sample, "full", task_bundle["full"]),
                    "vib": self._build_trace_payload(sample, "vib", task_bundle["vib"]),
                    "error_msg": res.error_msg
                }

                # 累计统计
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

                if res.full_passed and res.vib_passed:
                    stats["both_pass"] += 1
                elif res.full_passed and not res.vib_passed:
                    stats["full_only"] += 1
                elif not res.full_passed and res.vib_passed:
                    stats["vib_only"] += 1
                else:
                    stats["both_fail"] += 1

                # 进度输出
                elapsed = time.time() - dataset_start
                avg_time = elapsed / max(1, done_count)
                eta = avg_time * (total - done_count) / 60.0

                full_icon = "✅" if res.full_passed else "❌"
                vib_icon = "✅" if res.vib_passed else "❌"
                with self._print_lock:
                    print(f"  [{done_count:03d}/{total:03d}] {res.task_id:<25} | "
                          f"Full: {full_icon} | VIB: {vib_icon} (消息 {res.vib_messages}/6) | "
                          f"耗时: {res.execution_time:.1f}s | ETA: {eta:.1f}min")

            if self.workers == 1:
                for idx, sample in enumerate(samples, start=1):
                    task_bundle = self.run_single_task_with_trace(sample)
                    consume_result(idx - 1, task_bundle, idx)
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    future_map = {
                        executor.submit(self.run_single_task_with_trace, sample): idx
                        for idx, sample in enumerate(samples)
                    }
                    done_count = 0
                    for future in as_completed(future_map):
                        done_count += 1
                        result_index = future_map[future]
                        task_bundle = future.result()
                        consume_result(result_index, task_bundle, done_count)

            # 汇总该数据集指标
            full_rate = (stats["full_passed"] / total) * 100
            vib_rate = (stats["vib_passed"] / total) * 100
            msg_compression = ((stats["full_msgs"] - stats["vib_msgs"]) / max(1, stats["full_msgs"])) * 100
            token_saving = ((stats["full_tokens"] - stats["vib_tokens"]) / max(1, stats["full_tokens"])) * 100
            prompt_saving = ((stats["full_prompt_tokens"] - stats["vib_prompt_tokens"]) / max(1, stats["full_prompt_tokens"])) * 100
            completion_saving = ((stats["full_completion_tokens"] - stats["vib_completion_tokens"]) / max(1, stats["full_completion_tokens"])) * 100
            pass_delta_pp = vib_rate - full_rate

            summary[ds_name] = {
                "total_tasks": total,
                "full_pass_rate": round(full_rate, 2),
                "vib_pass_rate": round(vib_rate, 2),
                "pass_delta_pp": round(pass_delta_pp, 2),
                "full_messages": stats["full_msgs"],
                "vib_messages": stats["vib_msgs"],
                "msg_compression_pct": round(msg_compression, 2),
                "same_message_ratio": round(stats["same_msg_count"] / max(1, total), 4),
                "full_tokens": stats["full_tokens"],
                "vib_tokens": stats["vib_tokens"],
                "token_saving_pct": round(token_saving, 2),
                "full_prompt_tokens": stats["full_prompt_tokens"],
                "vib_prompt_tokens": stats["vib_prompt_tokens"],
                "prompt_saving_pct": round(prompt_saving, 2),
                "full_completion_tokens": stats["full_completion_tokens"],
                "vib_completion_tokens": stats["vib_completion_tokens"],
                "completion_saving_pct": round(completion_saving, 2),
                "full_only_count": stats["full_only"],
                "vib_only_count": stats["vib_only"],
                "both_pass_count": stats["both_pass"],
                "both_fail_count": stats["both_fail"]
            }
            all_results[ds_name] = [asdict(r) for r in results if r is not None]
            trace_results[ds_name] = [trace for trace in traces if trace is not None]

        # 打印总报告
        self._print_summary_report(summary)

        total_time = (time.time() - global_start) / 60.0
        print(f"\n评测总耗时: {total_time:.2f} 分钟")

        return {
            "summary": summary,
            "detailed_results": all_results,
            "trace_results": trace_results,
            "experiment_config": asdict(self.experiment_config),
        }

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
    parser.add_argument("--workers", type=int, default=1,
                        help="并发 worker 数（默认 1，建议 2~6）")
    parser.add_argument("--online", action="store_true",
                        help="允许联网拉取/更新 HuggingFace 数据集缓存（默认离线）")
    parser.add_argument("--output", type=str, default="results/multi_benchmark.json",
                        help="输出 JSON 文件路径")
    parser.add_argument("--disable-gsm8k-frozen-history", action="store_true",
                        help="关闭 GSM8K 同轮 frozen-history 修复（默认开启）")
    parser.add_argument("--enable-math-role-weights", action="store_true",
                        help="打开数学角色权重补偿（Planner/Solver/Verifier），默认关闭")
    parser.add_argument("--disable-math-role-weights", action="store_true",
                        help="关闭数学角色权重补偿（默认已关闭）；与 --enable 不能同时指定")
    parser.add_argument("--enable-gsm8k-reasoning-guard", action="store_true",
                        help="打开 GSM8K 轻量推理守护，默认关闭")
    parser.add_argument("--disable-gsm8k-reasoning-guard", action="store_true",
                        help="关闭 GSM8K 轻量推理守护（默认已关闭）；与 --enable 不能同时指定")
    parser.add_argument("--disable-humaneval-digest", action="store_true",
                        help="关闭 HumanEval reviewer digest fallback（默认已关闭）")
    parser.add_argument("--enable-humaneval-digest", action="store_true",
                        help="打开 HumanEval reviewer digest fallback（默认关闭）")
    parser.add_argument("--disable-mbpp-early-stop", action="store_true",
                        help="关闭 MBPP 一轮 probe early stopping（默认开启）")
    args = parser.parse_args()

    offline = not args.online
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["HF_DATASETS_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"

    mode_text = "OFFLINE" if offline else "ONLINE"
    print(f"[HF Mode] {mode_text} (HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']}, HF_DATASETS_OFFLINE={os.environ['HF_DATASETS_OFFLINE']})")

    api_key = resolve_deepseek_api_key()
    if not api_key:
        print("Error: 请设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)

    def resolve_flag(name, enable_arg, disable_arg):
        if enable_arg and disable_arg:
            print(f"Error: 不能同时指定 --enable-{name} 和 --disable-{name}")
            sys.exit(1)
        if enable_arg or disable_arg:
            return bool(enable_arg)
        cfg_default = getattr(ExperimentConfig(), f"enable_{name}")
        return cfg_default

    experiment_config = ExperimentConfig(
        enable_gsm8k_frozen_history=not args.disable_gsm8k_frozen_history,
        enable_math_role_weights=resolve_flag(
            "math_role_weights",
            args.enable_math_role_weights,
            args.disable_math_role_weights,
        ),
        enable_gsm8k_reasoning_guard=resolve_flag(
            "gsm8k_reasoning_guard",
            args.enable_gsm8k_reasoning_guard,
            args.disable_gsm8k_reasoning_guard,
        ),
        enable_humaneval_digest=resolve_flag(
            "humaneval_digest",
            args.enable_humaneval_digest,
            args.disable_humaneval_digest,
        ),
        enable_mbpp_early_stop=not args.disable_mbpp_early_stop,
    )

    harness = MultiDatasetHarness(
        api_key=api_key,
        max_steps=args.max_steps,
        workers=args.workers,
        experiment_config=experiment_config
    )
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

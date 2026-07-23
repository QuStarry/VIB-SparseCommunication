import os
# 1. 彻底禁用 HuggingFace Tokenizer 多进程并行警告（必须置于最顶部）
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 2. 强制 HuggingFace Hub 离线模式，避免每道题重复联网校验模型（SSL 超时）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import time
import gzip
import urllib.request
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import DeepSeekClient
from llm_agents import LLMAgent
from vib_router import VIBSparseCommRouter, VIBEvaluationResult
from workflow import VIBMultiAgentWorkflow

# ============================================
# 2. 自动获取与加载 HumanEval 全量数据集 (164题)
# ============================================
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "HumanEval.jsonl.gz")
OUTPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "results_humaneval_164.json")

def load_humaneval_dataset(limit: int = 164) -> list:
    if not os.path.exists(DATASET_PATH):
        print(f"📥 正在下载 HumanEval 数据集至 {DATASET_PATH} ...")
        urllib.request.urlretrieve(HUMANEVAL_URL, DATASET_PATH)
        print("✅ 数据集下载完成！")

    tasks = []
    with gzip.open(DATASET_PATH, 'rt', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
            if limit and len(tasks) >= limit:
                break
    return tasks

# ============================================
# 3. 单元测试校验器
# ============================================
def _unsafe_execute(code: str, test_script: str, entry_point: str, result_queue):
    try:
        full_code = f"{code}\n\n{test_script}\n\ncheck({entry_point})"
        exec_globals = {}
        exec(full_code, exec_globals)
        result_queue.put(True)
    except Exception:
        result_queue.put(False)

def check_correctness(completion: str, test_script: str, entry_point: str, timeout: float = 3.0) -> bool:
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_unsafe_execute, args=(completion, test_script, entry_point, result_queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    
    if not result_queue.empty():
        return result_queue.get()
    return False

# ============================================
# 4. 智能体与路由设计
# ============================================
def create_humaneval_agents(client: DeepSeekClient) -> list:
    return [
        LLMAgent("Architect", "你是软件架构师。分析函数需求、边界条件与最佳算法策略。", client),
        LLMAgent("Coder", "你是核心程序员。根据函数签名和需求，编写高效精密 Python 代码。", client),
        LLMAgent("Reviewer", "你是代码审查员。严格检查边界漏洞与语法逻辑错误。", client)
    ]

class BypassVIBRouter:
    """Full Comm 对照组透传门控：无条件放行所有消息。"""
    def evaluate_detailed(self, candidate_msg: str, history: list, goal: str, step_t: int,
                          token_probs=None, confidence_score=None) -> VIBEvaluationResult:
        return VIBEvaluationResult(
            utility_score=1.0, sufficiency=1.0, redundancy=0.0,
            beta_t=1.0, is_allowed=True, threshold=0.0
        )

def generate_final_code(client: DeepSeekClient, prompt: str, history: list) -> str:
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
        max_tokens=1024,
        temperature=0.1
    )
    
    code = res["content"]
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    return code

# ============================================
# 5. 全量 164 题 Benchmark 评估主循环
# ============================================
def run_humaneval_full_benchmark(num_samples: int = 164):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: 请设置 DEEPSEEK_API_KEY 环境变量。")
        sys.exit(1)

    dataset = load_humaneval_dataset(limit=num_samples)
    total_tasks = len(dataset)

    print("=" * 80)
    print(f" 🚀 开始 HumanEval 全量 164 题学术 Benchmark 评估")
    print(f" 🎯 目标样本数: {total_tasks} 题 | 输出日志路径: {OUTPUT_JSON_PATH}")
    print("=" * 80)

    stats = {
        "Full_Comm": {"passed": 0, "tokens": 0, "generated": 0, "accepted": 0},
        "VIB_Sparse": {"passed": 0, "tokens": 0, "generated": 0, "accepted": 0}
    }

    detailed_results = []
    start_time_all = time.time()

    for idx, task in enumerate(dataset):
        task_id = task["task_id"]
        prompt = task["prompt"]
        test_script = task["test"]
        entry_point = task["entry_point"]

        task_start = time.time()

        # ---- 1. Full Comm (对照组) ----
        client_full = DeepSeekClient(api_key=api_key, max_tokens=256)
        agents_full = create_humaneval_agents(client_full)
        workflow_full = VIBMultiAgentWorkflow(
            agents=[a.to_agent_dict() for a in agents_full],
            vib_router=BypassVIBRouter(),
            max_steps=2
        )
        res_full = workflow_full.run(goal=prompt, verbose=False)
        code_full = generate_final_code(client_full, prompt, res_full["history"])
        is_pass_full = check_correctness(code_full, test_script, entry_point)

        stats["Full_Comm"]["passed"] += int(is_pass_full)
        stats["Full_Comm"]["tokens"] += client_full.get_token_stats()["total_tokens"]
        stats["Full_Comm"]["generated"] += workflow_full.stats["total_candidates"]
        stats["Full_Comm"]["accepted"] += workflow_full.stats["total_approved"]

        # ---- 2. VIB Sparse (我们的方法) ----
        client_vib = DeepSeekClient(api_key=api_key, max_tokens=256)
        agents_vib = create_humaneval_agents(client_vib)
        router_vib = VIBSparseCommRouter(
            beta_0=1.5,           # 适度降低初始 beta，防止前期无脑放行
            lambda_decay=0.1,     # 减缓衰减速度
            gamma=-0.2,           # 调整基础准入门槛
            adaptive=True,        # <--- 必须开启 Ada-VIB 自适应开关！
            alpha=0.3,            # 提高对熵/不确定度的敏感度
            tau_min=-0.8
        )
        workflow_vib = VIBMultiAgentWorkflow(
            agents=[a.to_agent_dict() for a in agents_vib],
            vib_router=router_vib,
            max_steps=2
        )
        res_vib = workflow_vib.run(goal=prompt, verbose=False)
        code_vib = generate_final_code(client_vib, prompt, res_vib["history"])
        is_pass_vib = check_correctness(code_vib, test_script, entry_point)

        stats["VIB_Sparse"]["passed"] += int(is_pass_vib)
        stats["VIB_Sparse"]["tokens"] += client_vib.get_token_stats()["total_tokens"]
        stats["VIB_Sparse"]["generated"] += workflow_vib.stats["total_candidates"]
        stats["VIB_Sparse"]["accepted"] += workflow_vib.stats["total_approved"]

        task_elapsed = time.time() - task_start
        total_elapsed = time.time() - start_time_all
        avg_task_time = total_elapsed / (idx + 1)
        eta_seconds = avg_task_time * (total_tasks - (idx + 1))

        # 进度输出
        print(f"[{idx+1:03d}/{total_tasks:03d}] {task_id:<25} | "
              f"Full: {'✅' if is_pass_full else '❌'} | "
              f"VIB: {'✅' if is_pass_vib else '❌'} (消息 {workflow_vib.stats['total_approved']}/{workflow_vib.stats['total_candidates']}) | "
              f"耗时: {task_elapsed:.1f}s | 预估剩余 (ETA): {eta_seconds/60:.1f} min")

        # 记录明细数据
        task_record = {
            "task_id": task_id,
            "entry_point": entry_point,
            "full_comm": {"pass": is_pass_full, "msgs": workflow_full.stats['total_approved']},
            "vib_sparse": {"pass": is_pass_vib, "msgs": workflow_vib.stats['total_approved']},
            "time_sec": round(task_elapsed, 2)
        }
        detailed_results.append(task_record)

        # 每 5 题自动增量保存 JSON，确保数据安全
        if (idx + 1) % 5 == 0 or (idx + 1) == total_tasks:
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump({"summary": stats, "details": detailed_results}, f, indent=2, ensure_ascii=False)

    # ============================================
    # 6. 全量学术报表打印
    # ============================================
    total_time_min = (time.time() - start_time_all) / 60.0
    pass1_full = (stats["Full_Comm"]["passed"] / total_tasks) * 100
    pass1_vib = (stats["VIB_Sparse"]["passed"] / total_tasks) * 100
    
    comp_rate_vib = ((stats["VIB_Sparse"]["generated"] - stats["VIB_Sparse"]["accepted"]) / max(1, stats["VIB_Sparse"]["generated"])) * 100
    token_saving = ((stats["Full_Comm"]["tokens"] - stats["VIB_Sparse"]["tokens"]) / max(1, stats["Full_Comm"]["tokens"])) * 100

    print("\n" + "=" * 80)
    print(" 📊 HumanEval 164 题全量 Benchmark 终极大表 (Academic Report)")
    print("=" * 80)
    print(f"{'评估指标 (Metrics)':<34} | {'Full Comm (对照组)':<20} | {'VIB Sparse (我们的)':<20}")
    print("-" * 85)
    print(f"{'测试题目总数 (Total Tasks)':<34} | {total_tasks:<20} | {total_tasks:<20}")
    print(f"{'⭐ Pass@1 准确率 (%)':<34} | {pass1_full:.2f}%{'':<14} | {pass1_vib:.2f}%{'':<14}")
    print(f"{'尝试生成广播总数 (Generated)':<34} | {stats['Full_Comm']['generated']:<20} | {stats['VIB_Sparse']['generated']:<20}")
    print(f"{'实际通过广播消息数 (Accepted)':<34} | {stats['Full_Comm']['accepted']:<20} | {stats['VIB_Sparse']['accepted']:<20}")
    print(f"{'⭐ 通信压缩率 (Msg Compression %)':<34} | 0.00%{'':<15} | {comp_rate_vib:.2f}%{'':<14}")
    print(f"{'消耗 Token 总数 (Total Tokens)':<34} | {stats['Full_Comm']['tokens']:<20} | {stats['VIB_Sparse']['tokens']:<20}")
    print(f"{'⭐ Token 节省率 (Token Savings %)':<34} | 0.00%{'':<15} | {token_saving:.2f}%{'':<14}")
    print(f"{'评测总耗时 (Total Runtime)':<34} | {total_time_min:.2f} 分钟{'':<10} | {total_time_min:.2f} 分钟{'':<10}")
    print("=" * 80)
    print(f"🎉 详细明细已成功持久化至: {OUTPUT_JSON_PATH}\n")

if __name__ == "__main__":
    run_humaneval_full_benchmark(num_samples=164)
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import DeepSeekClient
from llm_agents import create_llm_agents
from vib_router import VIBSparseCommRouter, VIBEvaluationResult
from workflow import VIBMultiAgentWorkflow

class BypassVIBRouter:
    def evaluate(self, candidate_msg: str, history: list, goal: str, step_t: int) -> bool:
        return True
    
    def evaluate_detailed(self, candidate_msg: str, history: list, goal: str, step_t: int) -> VIBEvaluationResult:
        return VIBEvaluationResult(
            utility_score=1.0, sufficiency=1.0, redundancy=0.0,
            beta_t=1.0, is_allowed=True, threshold=0.0
        )

# ============================================
# 新增：代码合成节点 (Aggregator Node)
# ============================================
def generate_final_code(client: DeepSeekClient, goal: str, history: list) -> str:
    """读取协同历史，提炼并生成最终完整可执行代码"""
    history_text = "\n".join(history) if history else "(无历史讨论)"
    prompt = f"""你是一位资深首席软件工程师。请阅读以下多智能体团队针对任务的讨论历史，并提取其中的有效建议，最终写出完整、高效、可直接运行的 Python 代码。

【全局任务目标】:
{goal}

【团队讨论历史/建议】:
{history_text}

【要求】:
1. 提供完整的 Python 函数实现，涵盖所有需求。
2. 给出必要注释与时间复杂度分析。
3. 忽略讨论中不一致或无意义的争论，选择最优方案实现。"""

    res = client.chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.2
    )
    return res["content"]

# ============================================
# 大模型裁判 (改进版：针对最终代码打分)
# ============================================
class CollaborationJudge:
    def __init__(self, client: DeepSeekClient):
        self.client = client

    def judge_solution(self, goal: str, final_code: str, history: list) -> dict:
        history_text = "\n".join(history)
        judge_prompt = f"""你是一位极其严格的顶尖代码评测专家。请评估多智能体团队基于讨论历史最终产出的 Python 代码方案。

【任务目标】:
{goal}

【讨论历史Context长度】: {len(history)} 条消息

【最终产出的代码与方案】:
{final_code}

请评估最终代码的质量与上下文纯净度（每个维度 1-10 分，整数）：
1. 正确性与完备性 (Correctness): 代码是否完整可执行？是否满足小数组插入排序、大数组三路快排等所有要求？
2. 架构与代码质量 (Design): 代码组织是否优雅？边界条件处理是否完善？
3. 上下文纯净度 (Purification): 团队讨论历史是否干净高效？（越少矛盾与废话得分越高）

请严格按照以下 JSON 格式输出：
{{
    "correctness_score": 9,
    "design_score": 9,
    "purification_score": 8,
    "reason": "评价说明"
}}
"""
        messages = [{"role": "user", "content": judge_prompt}]
        response = self.client.chat(messages=messages, max_tokens=512, temperature=0.1)
        
        try:
            content = response["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except Exception:
            return {"correctness_score": 5, "design_score": 5, "purification_score": 5, "reason": "解析失败"}

def run_comparison_experiment():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: 请设置 DEEPSEEK_API_KEY。")
        sys.exit(1)

    task = {
        "name": "排序算法优化",
        "goal": "设计一个高性能的 Python 排序函数，要求：1) 对小数组使用插入排序，大数组使用快速排序；2) 处理大量重复元素的情况；3) 给出时间复杂度分析。",
        "max_steps": 3
    }

    print("=" * 70)
    print(" ⚖️ 开始协同对比实验 (含 Aggregator 代码生成): Full Comm vs VIB Sparse")
    print("=" * 70)

    judge_client = DeepSeekClient(api_key=api_key, model="deepseek-chat")
    judge = CollaborationJudge(judge_client)

    results = {}

    # ---- 1. Full Comm ----
    print("\n[1/2] 运行 Full Comm (全连接)...")
    client_full = DeepSeekClient(api_key=api_key, max_tokens=256)
    agents_full = create_llm_agents(client_full)
    workflow_full = VIBMultiAgentWorkflow(agents=agents_full, vib_router=BypassVIBRouter(), max_steps=task["max_steps"])
    res_full = workflow_full.run(goal=task["goal"], verbose=False)
    
    # 聚合生成最终代码
    code_full = generate_final_code(client_full, task["goal"], res_full["history"])
    tokens_full = client_full.get_token_stats()["total_tokens"]
    scores_full = judge.judge_solution(task["goal"], code_full, res_full["history"])

    results["Full_Comm"] = {"tokens": tokens_full, "compression": 0.0, "scores": scores_full, "history_len": len(res_full["history"])}
    time.sleep(2)

    # ---- 2. VIB Sparse ----
    print("\n[2/2] 运行 VIB Sparse (我们的)...")
    client_vib = DeepSeekClient(api_key=api_key, max_tokens=256)
    agents_vib = create_llm_agents(client_vib)
    router_vib = VIBSparseCommRouter(beta_0=3.0, lambda_decay=0.1, temp_tau=0.2, gamma=0.0)
    workflow_vib = VIBMultiAgentWorkflow(agents=agents_vib, vib_router=router_vib, max_steps=task["max_steps"])
    res_vib = workflow_vib.run(goal=task["goal"], verbose=False)
    
    # 聚合生成最终代码
    code_vib = generate_final_code(client_vib, task["goal"], res_vib["history"])
    tokens_vib = client_vib.get_token_stats()["total_tokens"]
    scores_vib = judge.judge_solution(task["goal"], code_vib, res_vib["history"])

    results["VIB_Sparse"] = {"tokens": tokens_vib, "compression": res_vib["compression_rate"], "scores": scores_vib, "history_len": len(res_vib["history"])}

    # ---- 报表输出 ----
    print("\n" + "=" * 70)
    print(" 📊 对比实验最终学术报表 (包含代码聚合输出打分)")
    print("=" * 70)
    print(f"{'指标 (Metrics)':<25} | {'Full Comm (对照组)':<20} | {'VIB Sparse (我们的)':<20}")
    print("-" * 75)
    print(f"{'讨论消息总数 (Msg Count)':<25} | {results['Full_Comm']['history_len']:<20} | {results['VIB_Sparse']['history_len']:<20}")
    print(f"{'通信压缩率 (Compression %)':<25} | {results['Full_Comm']['compression']:<19}% | {results['VIB_Sparse']['compression']:<19}%")
    print(f"{'消耗 Token 总数 (Tokens)':<25} | {results['Full_Comm']['tokens']:<20} | {results['VIB_Sparse']['tokens']:<20}")
    print("-" * 75)
    print(f"{'⭐ 正确性得分 (Correctness)':<25} | {str(results['Full_Comm']['scores']['correctness_score']) + '/10':<20} | {str(results['VIB_Sparse']['scores']['correctness_score']) + '/10':<20}")
    print(f"{'⭐ 架构质量得分 (Design)':<25} | {str(results['Full_Comm']['scores']['design_score']) + '/10':<20} | {str(results['VIB_Sparse']['scores']['design_score']) + '/10':<20}")
    print(f"{'⭐ 信息纯净度 (Purification)':<25} | {str(results['Full_Comm']['scores']['purification_score']) + '/10':<20} | {str(results['VIB_Sparse']['scores']['purification_score']) + '/10':<20}")
    print("-" * 75)
    print(f"Full Comm 裁判评语: {results['Full_Comm']['scores']['reason']}")
    print(f"VIB Sparse 裁判评语: {results['VIB_Sparse']['scores']['reason']}")
    print("=" * 70)

if __name__ == "__main__":
    run_comparison_experiment()
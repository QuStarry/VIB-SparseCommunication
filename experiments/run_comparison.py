import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import DeepSeekClient
from llm_agents import create_llm_agents
from vib_router import VIBSparseCommRouter, VIBEvaluationResult
from workflow import VIBMultiAgentWorkflow

# ============================================
# 1. 定义全连接对照组的“透明门控” (Bypass VIB)
# ============================================
class BypassVIBRouter:
    """模拟全连接通信：对所有候选消息百分之百放行，不进行任何剪枝"""
    def evaluate(self, candidate_msg: str, history: list, goal: str, step_t: int) -> bool:
        return True
    
    def evaluate_detailed(self, candidate_msg: str, history: list, goal: str, step_t: int) -> VIBEvaluationResult:
        return VIBEvaluationResult(
            utility_score=1.0, sufficiency=1.0, redundancy=0.0,
            beta_t=1.0, is_allowed=True, threshold=0.0
        )

# ============================================
# 2. 定义大模型裁判 (LLM Judge)
# ============================================
class CollaborationJudge:
    def __init__(self, client: DeepSeekClient):
        self.client = client

    def judge_solution(self, goal: str, final_history: list) -> dict:
        """从三个维度对多智能体最终协同出的方案进行盲审打分"""
        history_text = "\n".join(final_history)
        
        judge_prompt = f"""你是一位极其严格的顶尖代码评测专家。请对以下多智能体协同协作的最终结果进行深度评估。

【任务目标】:
{goal}

【多智能体协作全文本/最终方案】:
{history_text}

请严格根据上述文本，从以下三个维度进行打分（每个维度 1-10 分，分数必须为整数）：
1. 正确性与完备性 (Correctness): 是否完整实现了目标的所有要求？代码是否有潜在Bug或边界漏洞？
2. 架构与代码质量 (Design): 代码是否优雅？是否符合设计模式和最佳实践？可扩展性如何？
3. 信息纯净度 (Purification): 最终讨论结果是否存在大量重复的废话、互相恭维或无意义的噪声？（噪声越多，此项得分越低）

请务必严格按照以下 JSON 格式输出，不要有任何多余的解释：
{{
    "correctness_score": 8,
    "design_score": 9,
    "purification_score": 7,
    "reason": "简短的一句评价原因"
}}
"""
        messages = [{"role": "user", "content": judge_prompt}]
        response = self.client.chat(messages=messages, max_tokens=512, temperature=0.1) # 低温保证客观性
        
        try:
            # 提取 JSON
            content = response["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except Exception:
            # 降级解析
            return {"correctness_score": 5, "design_score": 5, "purification_score": 5, "reason": "解析裁判输出失败"}

# ============================================
# 3. 核心对比测试流程
# ============================================
def run_comparison_experiment():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: 请先设置 DEEPSEEK_API_KEY 环境变量。")
        sys.exit(1)

    # 评测任务（选用你之前跑过的标准排序优化任务）
    task = {
        "name": "排序算法优化",
        "goal": "设计一个高性能的Python排序函数，要求：1) 对小数组使用插入排序，大数组使用快速排序；2) 处理大量重复元素的情况；3) 给出时间复杂度分析。",
        "max_steps": 4
    }

    print("=" * 70)
    print(" ⚖️ 开始进行多智能体协同对比实验：Full Comm vs VIB Sparse")
    print("=" * 70)

    # ---- 实例化评测裁判 ----
    judge_client = DeepSeekClient(api_key=api_key, model="deepseek-chat")
    judge = CollaborationJudge(judge_client)

    results = {}

    # ---- 实验组 1: 全连接 (Full Communication) ----
    print("\n运行 [1/2] 对照组：Full Communication (无过滤全连接)...")
    client_full = DeepSeekClient(api_key=api_key, max_tokens=256)
    agents_full = create_llm_agents(client_full)
    workflow_full = VIBMultiAgentWorkflow(agents=agents_full, vib_router=BypassVIBRouter(), max_steps=task["max_steps"])
    
    res_full = workflow_full.run(goal=task["goal"], verbose=False)
    tokens_full = client_full.get_token_stats()["total_tokens"]
    scores_full = judge.judge_solution(task["goal"], res_full["history"])

    results["Full_Comm"] = {"tokens": tokens_full, "compression": 0.0, "scores": scores_full, "history_len": len(res_full["history"])}
    time.sleep(2) # 避免触发 Rate Limit

    # ---- 实验组 2: VIB 稀疏通信 (VIB Sparse Comm) ----
    print("\n运行 [2/2] 实验组：VIB-SparseComm (我们的动态稀疏过滤)...")
    client_vib = DeepSeekClient(api_key=api_key, max_tokens=256)
    agents_vib = create_llm_agents(client_vib)
    # 使用消融实验表现优秀的参数组
    router_vib = VIBSparseCommRouter(beta_0=3.0, lambda_decay=0.1, temp_tau=0.2, gamma=0.0)
    workflow_vib = VIBMultiAgentWorkflow(agents=agents_vib, vib_router=router_vib, max_steps=task["max_steps"])
    
    res_vib = workflow_vib.run(goal=task["goal"], verbose=False)
    tokens_vib = client_vib.get_token_stats()["total_tokens"]
    scores_vib = judge.judge_solution(task["goal"], res_vib["history"])

    results["VIB_Sparse"] = {"tokens": tokens_vib, "compression": res_vib["compression_rate"], "scores": scores_vib, "history_len": len(res_vib["history"])}

    # ============================================
    # 4. 打印学术对比报表
    # ============================================
    print("\n" + "=" * 70)
    print(" 📊 对比实验最终学术报表 (Comparison Report)")
    print("=" * 70)
    print(f"{'指标 (Metrics)':<25} | {'Full Comm (对照组)':<20} | {'VIB Sparse (我们的)':<20}")
    print("-" * 75)
    print(f"{'广播消息总数 (Msg Count)':<25} | {results['Full_Comm']['history_len']:<20} | {results['VIB_Sparse']['history_len']:<20}")
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
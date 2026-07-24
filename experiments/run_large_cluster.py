import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import DeepSeekClient
from llm_agents import LLMAgent
from vib_router import VIBSparseCommRouter, VIBEvaluationResult
from workflow import VIBMultiAgentWorkflow

# ============================================
# 1. 打造 8 智能体大集群 (8-Agent Cluster)
# 采用位置参数避免不同类定义的关键字不一致问题
# ============================================
def create_8_agent_cluster(client: DeepSeekClient) -> list:
    """创建包含 8 个不同职责的扩展智能体集群，返回 workflow 兼容的 dict 列表"""
    agents = [
        LLMAgent(
            "Architect",
            "你是首席架构师。关注系统整体架构、数据结构选择与模块解耦。",
            client
        ),
        LLMAgent(
            "Programmer",
            "你是核心程序员。负责根据设计要求编写高效、干净的 Python 代码。",
            client
        ),
        LLMAgent(
            "Code_Reviewer",
            "你是代码审查员。关注代码规范、潜在 Bug 和异常处理。",
            client
        ),
        LLMAgent(
            "Security_Auditor",
            "你是安全审计员。关注内存泄漏、线程安全、并发死锁以及边界碰撞防护。",
            client
        ),
        LLMAgent(
            "Performance_Optimizer",
            "你是性能优化师。极度追求时间/空间复杂度，关注 Python 内置数据结构的底层开销。",
            client
        ),
        LLMAgent(
            "QA_Tester",
            "你是测试工程师。关注边界条件（如空值、极大值、并发竞争）和单元测试覆盖率。",
            client
        ),
        LLMAgent(
            "Refactoring_Specialist",
            "你是重构专家。关注代码可读性、设计模式复用与DRY原则。",
            client
        ),
        LLMAgent(
            "Documentation_Specialist",
            "你是文档专员。关注 Type Hints、Docstring 格式与 API 说明文档。",
            client
        )
    ]
    # 转换为 workflow 兼容的 dict 格式 {"name": ..., "generate": ...}
    return [agent.to_agent_dict() for agent in agents]

# ============================================
# 2. 全连接透传门控
# ============================================
class BypassVIBRouter:
    def evaluate(self, candidate_msg: str, history: list, goal: str, step_t: int) -> bool:
        return True
    
    def evaluate_detailed(self, candidate_msg: str, history: list, goal: str, step_t: int,
                          token_probs=None, confidence_score=None, role=None) -> VIBEvaluationResult:
        return VIBEvaluationResult(
            utility_score=1.0, sufficiency=1.0, redundancy=0.0,
            beta_t=1.0, is_allowed=True, threshold=0.0
        )

# ============================================
# 3. 代码聚合生成与 LLM 裁判
# ============================================
def generate_final_code(client: DeepSeekClient, goal: str, history: list) -> str:
    history_text = "\n".join(history) if history else "(无历史讨论)"
    prompt = f"""你是一位首席工程师。请阅读 8 人团队针对复杂任务的讨论历史，屏蔽无意义的碎碎念与格式争论，提取真正有价值的架构建议，最终输出完整、高效、可执行的 Python 代码。

【任务目标】:
{goal}

【8人团队讨论历史】:
{history_text}

【要求】:
1. 提供完整可执行的 Python 代码，涵盖所有高级特性。
2. 处理好并发与边界防御。
3. 给出详细注释与复杂度分析。"""

    res = client.chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.2
    )
    return res["content"]

class LargeClusterJudge:
    def __init__(self, client: DeepSeekClient):
        self.client = client

    def judge_solution(self, goal: str, final_code: str, history: list) -> dict:
        judge_prompt = f"""你是一位极其严苛的分布式系统专家。请评估 8 人团队基于协同历史产出的最终代码。

【任务目标】:
{goal}

【历史 Context 消息数】: {len(history)} 条

【最终代码与方案】:
{final_code}

请严格评估（1-10 分整数）：
1. 正确性与完备性 (Correctness): 是否完整实现核心算法及并发/过期特性？代码是否无 Bug 直接运行？
2. 架构与代码质量 (Design): 面对 8 人复杂意见时，架构是否依然保持干净优雅？是否陷入过度工程？
3. 高密度噪声抗性 (Noise Resistance): 8 人大集群讨论中，最终历史的纯净度如何？是否成功阻断了格式争论等噪音？

请严格输出 JSON:
{{
    "correctness_score": 9,
    "design_score": 9,
    "noise_resistance_score": 8,
    "reason": "简要评价"
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
            return {"correctness_score": 5, "design_score": 5, "noise_resistance_score": 5, "reason": "解析裁判输出失败"}

# ============================================
# 4. 执行大集群高密度噪声实验
# ============================================
def run_large_cluster_experiment():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: 请设置 DEEPSEEK_API_KEY。")
        sys.exit(1)

    task = {
        "name": "线程安全的 LFU-TTL 缓存系统",
        "goal": "设计并实现一个高性能、线程安全的 Python LFU (Least Frequently Used) 缓存类，要求：1) 支持 O(1) 的 Get 和 Put 操作；2) 支持 TTL (生存时间) 自动过期淘汰；3) 具备死锁防护机制；4) 包含完整的单元测试。",
        "max_steps": 3
    }

    print("=" * 75)
    print(" 🚀 开始大集群扩展测试 (8-Agent Cluster Noise Resistance Test)")
    print("=" * 75)

    judge_client = DeepSeekClient(api_key=api_key, model="deepseek-chat")
    judge = LargeClusterJudge(judge_client)

    results = {}

    # ---- [1/2] 8-Agent Full Comm ----
    print("\n[1/2] 运行 8-Agent Full Comm (全连接大集群)...")
    client_full = DeepSeekClient(api_key=api_key, max_tokens=256)
    agents_full = create_8_agent_cluster(client_full)
    workflow_full = VIBMultiAgentWorkflow(agents=agents_full, vib_router=BypassVIBRouter(), max_steps=task["max_steps"])
    
    res_full = workflow_full.run(goal=task["goal"], verbose=False)
    code_full = generate_final_code(client_full, task["goal"], res_full["history"])
    tokens_full = client_full.get_token_stats()["total_tokens"]
    scores_full = judge.judge_solution(task["goal"], code_full, res_full["history"])

    results["Full_Comm_8"] = {
        "tokens": tokens_full,
        "compression": 0.0,
        "scores": scores_full,
        "history_len": len(res_full["history"])
    }
    time.sleep(3)

    # ---- [2/2] 8-Agent VIB SparseComm ----
    print("\n[2/2] 运行 8-Agent VIB-SparseComm (我们的动态门控大集群)...")
    client_vib = DeepSeekClient(api_key=api_key, max_tokens=256)
    agents_vib = create_8_agent_cluster(client_vib)
    router_vib = VIBSparseCommRouter(beta_0=3.5, lambda_decay=0.15, temp_tau=0.2, gamma=0.0)
    workflow_vib = VIBMultiAgentWorkflow(agents=agents_vib, vib_router=router_vib, max_steps=task["max_steps"])
    
    res_vib = workflow_vib.run(goal=task["goal"], verbose=False)
    code_vib = generate_final_code(client_vib, task["goal"], res_vib["history"])
    tokens_vib = client_vib.get_token_stats()["total_tokens"]
    scores_vib = judge.judge_solution(task["goal"], code_vib, res_vib["history"])

    results["VIB_Sparse_8"] = {
        "tokens": tokens_vib,
        "compression": res_vib["compression_rate"],
        "scores": scores_vib,
        "history_len": len(res_vib["history"])
    }

    # ============================================
    # 5. 学术报表 (使用独立格式化变量，绝对不报 f-string 错误)
    # ============================================
    full_corr = f"{results['Full_Comm_8']['scores']['correctness_score']}/10"
    vib_corr = f"{results['VIB_Sparse_8']['scores']['correctness_score']}/10"

    full_des = f"{results['Full_Comm_8']['scores']['design_score']}/10"
    vib_des = f"{results['VIB_Sparse_8']['scores']['design_score']}/10"

    full_noise = f"{results['Full_Comm_8']['scores']['noise_resistance_score']}/10"
    vib_noise = f"{results['VIB_Sparse_8']['scores']['noise_resistance_score']}/10"

    print("\n" + "=" * 75)
    print(" 📊 8-Agent 大集群高密度噪声测试学术报表")
    print("=" * 75)
    print(f"{'评估指标 (Metrics)':<28} | {'8-Agent Full Comm':<20} | {'8-Agent VIB Sparse':<20}")
    print("-" * 80)
    print(f"{'广播消息总量 (Msg Count)':<28} | {results['Full_Comm_8']['history_len']:<20} | {results['VIB_Sparse_8']['history_len']:<20}")
    print(f"{'通信压缩率 (Compression %)':<28} | {results['Full_Comm_8']['compression']:<19}% | {results['VIB_Sparse_8']['compression']:<19}%")
    print(f"{'消耗 Token 总数 (Tokens)':<28} | {results['Full_Comm_8']['tokens']:<20} | {results['VIB_Sparse_8']['tokens']:<20}")
    print("-" * 80)
    print(f"{'⭐ 代码正确性 (Correctness)':<28} | {full_corr:<20} | {vib_corr:<20}")
    print(f"{'⭐ 架构质量 (Design)':<28} | {full_des:<20} | {vib_des:<20}")
    print(f"{'⭐ 高密度抗噪性 (Noise Resist)':<28} | {full_noise:<20} | {vib_noise:<20}")
    print("-" * 80)
    print(f"Full Comm 裁判评语: {results['Full_Comm_8']['scores']['reason']}")
    print(f"VIB Sparse 裁判评语: {results['VIB_Sparse_8']['scores']['reason']}")
    print("=" * 75)

if __name__ == "__main__":
    run_large_cluster_experiment()
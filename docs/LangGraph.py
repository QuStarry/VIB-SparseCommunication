import math
import numpy as np
import operator
from typing import Annotated, List, Dict, TypedDict
from langgraph.graph import StateGraph, END

# ==========================================
# 1. 定义 LangGraph 全局共享状态 (State)
# ==========================================
class AgentState(TypedDict):
    global_goal: str                             # 全局任务目标
    history: Annotated[List[str], operator.add]  # 公共历史信息（通过累加器追加保存）
    candidates: List[Dict[str, str]]             # 临时缓冲区：存储各 Agent 本轮的候选发言
    step_t: int                                  # 当前协同轮次 t


# ==========================================
# 2. 变分信息瓶颈 (VIB) 评估器
# ==========================================
class VIBSparseCommRouter:
    def __init__(self, beta_0: float = 2.0, lambda_decay: float = 0.15, temp_tau: float = 0.2):
        self.beta_0 = beta_0
        self.lambda_decay = lambda_decay
        self.tau = temp_tau
        # 2026年主流的本地极速 Embedding 方案，用于计算互信息
        self.encoder = LightweightEmbeddingModel("bge-small-zh")

    def _cos_sim(self, v1, v2):
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

    def evaluate(self, draft: str, history: List[str], goal: str, t: int) -> bool:
        if not history:
            return True # 第一轮强制放行建立基调
            
        emb_m = self.encoder.encode(draft)
        emb_g = self.encoder.encode(goal)
        emb_h = [self.encoder.encode(h) for h in history]

        # 估算充分性 I(M; G)
        i_m_g = self._cos_sim(emb_m, emb_g)

        # 估算冗余度 I(M; H) (通过对比密度估计)
        sim_scores = [self._cos_sim(emb_m, h) for h in emb_h]
        i_m_h = math.log(sum(math.exp(s / self.tau) for s in sim_scores) + 1e-9)

        # 动态衰减拉格朗日乘子
        beta_t = self.beta_0 * math.exp(-self.lambda_decay * t)

        # 变分信息瓶颈评分
        utility = beta_t * i_m_g - i_m_h
        return utility >= 0.0


# ==========================================
# 3. 定义 LangGraph 节点 (Nodes)
# ==========================================
# 初始化全局 VIB 路由实例
vib_router = VIBSparseCommRouter()

def programmer_agent_node(state: AgentState):
    """程序员智能体：在本地生成代码或方案（不直接写入公共历史）"""
    prompt = f"目标: {state['global_goal']}\n当前公共历史: {state['history']}\n请给出你的修改建议。"
    # 调用大模型 (本地私有思考)
    local_thought = call_llm(prompt) 
    
    # 写入 candidates 缓冲区
    return {"candidates": [{"agent": "Programmer", "content": local_thought}]}


def reviewer_agent_node(state: AgentState):
    """审查员智能体：独立生成审查意见"""
    prompt = f"目标: {state['global_goal']}\n当前公共历史: {state['history']}\n请指出当前方案的潜在漏洞。"
    local_thought = call_llm(prompt)
    
    return {"candidates": [{"agent": "Reviewer", "content": local_thought}]}


def vib_gatekeeper_node(state: AgentState):
    """VIB 门控节点：决定哪些 candidates 可以晋升到全局 history 中"""
    approved_broadcasts = []
    
    for item in state["candidates"]:
        is_allowed = vib_router.evaluate(
            draft=item["content"],
            history=state["history"],
            goal=state["global_goal"],
            t=state["step_t"]
        )
        if is_allowed:
            approved_broadcasts.append(f"[{item['agent']}]: {item['content']}")
            
    # 返回更新：approved_broadcasts 会由于 operator.add 被自动追加到 history 中
    return {
        "history": approved_broadcasts,
        "candidates": [],               # 清空本轮缓冲区
        "step_t": state["step_t"] + 1  # 步数加 1
    }


# ==========================================
# 4. 构建 LangGraph 工作流拓扑
# ==========================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("programmer", programmer_agent_node)
workflow.add_node("reviewer", reviewer_agent_node)
workflow.add_node("vib_gatekeeper", vib_gatekeeper_node)

# 设置起点：同时触发多个 Agent 进行本地思考 (并行计算)
workflow.set_entry_point(["programmer", "reviewer"])

# 智能体思考完毕后，统一汇总到 VIB 门控节点
workflow.add_edge("programmer", "vib_gatekeeper")
workflow.add_edge("reviewer", "vib_gatekeeper")

# 设置门控节点的条件路由 (决定是继续下一轮迭代，还是直接结束)
def should_continue(state: AgentState):
    # 终止条件：任务解决，或者达到了最大迭代步数
    if is_task_completed(state["history"]) or state["step_t"] >= 8:
        return END
    # 否则，清空 candidates 后重新分发给各个 Agent 节点
    return ["programmer", "reviewer"]

workflow.add_conditional_edges("vib_gatekeeper", should_continue)

# 编译运行图
app = workflow.compile()
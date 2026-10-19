"""
LLM-based Agents for VIB-SparseComm Multi-Agent System

Implements real LLM-powered agents using DeepSeek API for multi-agent collaboration.
"""

from typing import List, Dict, Optional

try:
    from .llm_client import DeepSeekClient
except ImportError:
    from llm_client import DeepSeekClient


# ============================================
# Agent System Prompts
# ============================================

PROGRAMMER_SYSTEM_PROMPT = """你是一位资深Python程序员智能体。你的职责是针对给定的编程任务，提供具体的代码实现方案和优化建议。

要求：
1. 给出具体的、可执行的代码建议
2. 关注算法效率、代码质量和最佳实践
3. 每次发言控制在2-3句话内，聚焦于一个具体的技术点
4. 不要重复其他人已经说过的内容
5. 如果当前讨论已经足够完善，可以提出边缘案例或性能测试建议"""

REVIEWER_SYSTEM_PROMPT = """你是一位代码审查员智能体。你的职责是审查当前讨论中的方案，指出潜在问题并提出改进方向。

要求：
1. 关注方案的漏洞、边界条件和潜在风险
2. 提出具体的改进建议而非泛泛而谈
3. 每次发言控制在2-3句话内，聚焦于一个新的观点
4. 不要简单附和他人观点，要提供独特视角
5. 如果方案已经足够完善，可以总结关键要点"""

ARCHITECT_SYSTEM_PROMPT = """你是一位系统架构师智能体。你的职责是从更高层次审视问题，提供架构层面的思考和技术选型建议。

要求：
1. 关注整体设计模式、可扩展性和技术选型
2. 提供不同于程序员和审查员的宏观视角
3. 每次发言控制在2-3句话内，聚焦于一个架构决策
4. 引入新的技术维度或设计考量
5. 避免重复已有的讨论内容"""


class LLMAgent:
    """
    LLM-powered agent for multi-agent collaboration.
    
    Each agent has a specific role and generates responses using DeepSeek API.
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_client: DeepSeekClient,
        max_tokens: int = 256
    ):
        """
        Initialize LLM agent.
        
        Args:
            name: Agent name/role
            system_prompt: System prompt defining agent behavior
            llm_client: Shared DeepSeek client instance
            max_tokens: Max tokens per response (keep short for sparse comm)
        """
        self.name = name
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        
        # Track per-agent token usage
        self.tokens_used = 0
        self.calls_made = 0
    
    def generate(self, goal: str, history: List[str], step_t: int) -> str:
        """
        Generate a candidate message based on current context.
        
        Args:
            goal: Global task goal
            history: Current public communication history
            step_t: Current collaboration step
            
        Returns:
            Generated message content
        """
        # Build context from history
        history_text = "\n".join(history) if history else "(暂无历史讨论)"
        
        # Construct user message
        user_message = f"""【全局任务目标】: {goal}

【当前公共讨论历史】:
{history_text}

【当前轮次】: 第 {step_t} 轮

请基于以上信息，以你的角色视角给出你的贡献。注意：
- 只输出你的发言内容（2-3句话）
- 不要输出角色名或前缀
- 提供新的、有价值的信息，避免重复已有内容"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        result = self.llm_client.chat(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.2  # 修改点：从 0.7 降至 0.2，增强代码逻辑严谨度
        )
        
        # Track usage
        self.tokens_used += result["total_tokens"]
        self.calls_made += 1
        
        return (result["content"] or "").strip()
    
    def to_agent_dict(self) -> Dict:
        """Convert to agent dict format for workflow."""
        return {
            "name": self.name,
            "generate": self.generate
        }


def create_llm_agents(
    llm_client: DeepSeekClient,
    roles: Optional[List[str]] = None
) -> List[Dict]:
    """
    Create a set of LLM agents for collaboration.
    
    Args:
        llm_client: Shared DeepSeek client
        roles: List of roles to create (default: all three)
        
    Returns:
        List of agent dicts ready for VIBMultiAgentWorkflow
    """
    role_configs = {
        "Programmer": PROGRAMMER_SYSTEM_PROMPT,
        "Reviewer": REVIEWER_SYSTEM_PROMPT,
        "Architect": ARCHITECT_SYSTEM_PROMPT
    }
    
    if roles is None:
        roles = ["Programmer", "Reviewer", "Architect"]
    
    agents = []
    for role in roles:
        if role in role_configs:
            agent = LLMAgent(
                name=role,
                system_prompt=role_configs[role],
                llm_client=llm_client,
                max_tokens=256
            )
            agents.append(agent.to_agent_dict())
    
    return agents


def create_baseline_agent(llm_client: DeepSeekClient) -> Dict:
    """
    Create a single-agent baseline (no multi-agent, no VIB filtering).
    Used for comparison in ablation studies.
    """
    system_prompt = """你是一位全能型AI助手。请针对给定任务提供完整的解决方案。
要求：给出全面、详细的回答，涵盖实现方案、注意事项和优化建议。"""
    
    agent = LLMAgent(
        name="Baseline",
        system_prompt=system_prompt,
        llm_client=llm_client,
        max_tokens=2048
    )
    return agent.to_agent_dict()

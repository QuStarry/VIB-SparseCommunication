"""
LangGraph Multi-Agent Workflow with VIB Sparse Communication

This module implements the complete multi-agent collaboration system using LangGraph,
with VIB-based sparse communication gating.
"""

import operator
import re
from typing import Annotated, List, Dict, TypedDict, Callable, Optional
from langgraph.graph import StateGraph, END

try:
    from .vib_router import VIBSparseCommRouter, VIBEvaluationResult
except ImportError:
    from vib_router import VIBSparseCommRouter, VIBEvaluationResult


class AgentState(TypedDict):
    """
    Global shared state for the multi-agent collaboration system.
    
    Attributes:
        global_goal: The global task objective/instruction
        history: Public communication history (approved messages only)
        candidates: Temporary buffer for pending candidate messages
        step_t: Current collaboration round number
        total_candidates: Total number of candidates generated
        total_approved: Total number of candidates approved
    """
    global_goal: str
    history: Annotated[List[str], operator.add]
    candidates: List[Dict[str, str]]
    step_t: int
    total_candidates: int
    total_approved: int


class VIBMultiAgentWorkflow:
    """
    Multi-agent workflow with VIB-based sparse communication.
    
    This class builds and manages a LangGraph workflow where:
    1. Multiple agents generate candidate messages in parallel
    2. VIB router evaluates each candidate for information gain
    3. Only high-value messages are broadcast to shared history
    4. Process repeats until task completion or max steps
    """
    
    def __init__(
        self,
        agents: List[Dict[str, Callable]],
        vib_router: Optional[VIBSparseCommRouter] = None,
        max_steps: int = 8,
        completion_checker: Optional[Callable] = None,
        same_round_history_mode: str = "incremental",
        enable_digest_fallback: bool = False,
        required_roles_first_step: Optional[List[str]] = None,
        first_step_min_approvals: int = 0
    ):
        """
        Initialize the multi-agent workflow.
        
        Args:
            agents: List of agent dicts with 'name' and 'generate' function
            vib_router: VIB router instance (creates default if None)
            max_steps: Maximum collaboration rounds
            completion_checker: Function to check if task is complete
        """
        self.agents = agents
        self.vib_router = vib_router or VIBSparseCommRouter()
        self.max_steps = max_steps
        self.completion_checker = completion_checker or self._default_completion_checker
        self.same_round_history_mode = same_round_history_mode
        self.enable_digest_fallback = enable_digest_fallback
        self.required_roles_first_step = list(required_roles_first_step or [])
        self.first_step_min_approvals = first_step_min_approvals
        
        # Statistics tracking
        self.stats = {
            "total_candidates": 0,
            "total_approved": 0,
            "total_blocked": 0,
            "total_digests": 0,
            "total_role_rescues": 0,
            "steps_completed": 0,
            "history_per_step": [],
            "gate_decisions": []
        }
        
        # Build the workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _default_completion_checker(self, history: List[str], goal: str) -> bool:
        """Default completion checker - always returns False (runs to max_steps)."""
        return False
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow topology."""
        workflow = StateGraph(AgentState)
        
        # Add a coordinator node that runs all agents
        workflow.add_node("agents", self._agents_node)
        
        # Add VIB gatekeeper node
        workflow.add_node("vib_gatekeeper", self._vib_gatekeeper_node)
        
        # Set entry point
        workflow.set_entry_point("agents")
        
        # Connect agents to gatekeeper
        workflow.add_edge("agents", "vib_gatekeeper")
        
        # Add conditional edges from gatekeeper
        workflow.add_conditional_edges(
            "vib_gatekeeper",
            self._should_continue,
            {
                "continue": "agents",
                "end": END
            }
        )
        
        return workflow
    
    def _agents_node(self, state: AgentState) -> Dict:
        """Run all agents and collect their candidates."""
        all_candidates = []
        
        for agent in self.agents:
            agent_name = agent["name"]
            generate_fn = agent["generate"]
            
            # Generate candidate message
            content = generate_fn(
                goal=state["global_goal"],
                history=state["history"],
                step_t=state["step_t"]
            )
            
            all_candidates.append({"agent": agent_name, "content": content})
        
        return {"candidates": all_candidates}
    
    def _create_agent_node(self, agent: Dict) -> Callable:
        """Create a node function for an agent."""
        agent_name = agent["name"]
        generate_fn = agent["generate"]
        
        def agent_node(state: AgentState) -> Dict:
            # Generate candidate message using agent's function
            content = generate_fn(
                goal=state["global_goal"],
                history=state["history"],
                step_t=state["step_t"]
            )
            
            # Return candidate (will be added to candidates list)
            return {
                "candidates": [{"agent": agent_name, "content": content}]
            }
        
        return agent_node

    def _should_emit_digest(self, item: Dict[str, str], result: VIBEvaluationResult) -> bool:
        if not self.enable_digest_fallback:
            return False
        if item.get("agent") not in {"Reviewer", "Coder"}:
            return False
        if result.is_allowed:
            return False
        content = item.get("content", "")
        content_lower = content.lower()
        edge_keywords = [
            "edge", "edge case", "boundary", "corner", "corner case",
            "empty", "zero", "negative", "none", "null", "invalid",
            "exception", "overflow", "off-by-one", "special case",
            "fails when", "fails if", "fix", "bug", "missing"
        ]
        return any(keyword in content_lower for keyword in edge_keywords)

    def _build_digest_message(self, item: Dict[str, str]) -> str:
        content = " ".join(item.get("content", "").split())
        sentences = re.split(r"(?<=[。.!?])\s+", content)
        digest = sentences[0].strip() if sentences else content
        agent = item.get("agent", "Agent")
        return f"[{agent}-digest]: {digest[:220]}"

    def _rescue_required_roles(
        self,
        state: AgentState,
        blocked_items: List[Dict[str, str]],
        approved_broadcasts: List[str],
        running_history: List[str],
        approved_roles: set
    ) -> List[Dict[str, str]]:
        if state["step_t"] != 1 or not self.required_roles_first_step:
            return blocked_items

        remaining_blocked = list(blocked_items)
        target_approvals = max(self.first_step_min_approvals, 0)
        if target_approvals and len(approved_broadcasts) >= target_approvals:
            return remaining_blocked

        for role in self.required_roles_first_step:
            if target_approvals and len(approved_broadcasts) >= target_approvals:
                break
            if role in approved_roles:
                continue
            match = next((item for item in remaining_blocked if item.get("agent") == role), None)
            if match is None:
                continue
            formatted_msg = f"[{match['agent']}]: {match['content']}"
            approved_broadcasts.append(formatted_msg)
            running_history.append(formatted_msg)
            approved_roles.add(role)
            remaining_blocked.remove(match)
            self.stats["total_approved"] += 1
            self.stats["total_role_rescues"] += 1

        return remaining_blocked
    
    def _vib_gatekeeper_node(self, state: AgentState) -> Dict:
        """
        VIB Gatekeeper Node: 修复批处理内部历史同步问题
        """
        approved_broadcasts = []
        candidates = state.get("candidates", [])
        
        self.stats["total_candidates"] += len(candidates)
        
        # 建立当前轮次的增量历史视图
        running_history = list(state["history"]) # 浅拷贝当前全局历史
        approved_roles = set()
        blocked_items = []
        
        for item in candidates:
            history_view = running_history if self.same_round_history_mode == "incremental" else state["history"]
            # 实时传入包含同轮次已批准消息的 running_history
            result = self.vib_router.evaluate_detailed(
                candidate_msg=item["content"],
                history=history_view,
                goal=state["global_goal"],
                step_t=state["step_t"],
                token_probs=item.get("token_probs"), # 透传 Ada-VIB 所需的指标
                confidence_score=item.get("confidence"),
                role=item.get("agent"),  # 透传角色信息，启用 Role-Adaptive β
                is_execution_failed=state.get("execution_failed", False)  # 透传执行反馈信号
            )
            # === HIGH-1: serialize gate decision for paper §3.2 audit trail ===
            msg_snip = item["content"][:160].replace("\n", "\\n")
            self.stats["gate_decisions"].append({
                "round_idx": state["step_t"],
                "role": item.get("agent"),
                "utility_score": float(result.utility_score),
                "sufficiency": float(result.sufficiency),
                "redundancy": float(result.redundancy),
                "beta_t": float(result.beta_t),
                "threshold": float(result.threshold),
                "is_allowed": bool(result.is_allowed),
                "rescued": False,
                "digest": False,
                "msg_snip": msg_snip
            })
            gate_decision_idx = len(self.stats["gate_decisions"]) - 1

            if result.is_allowed:
                formatted_msg = f"[{item['agent']}]: {item['content']}"
                approved_broadcasts.append(formatted_msg)
                running_history.append(formatted_msg)
                self.stats["total_approved"] += 1
                approved_roles.add(item["agent"])
            elif self._should_emit_digest(item, result):
                digest_msg = self._build_digest_message(item)
                approved_broadcasts.append(digest_msg)
                running_history.append(digest_msg)
                self.stats["total_approved"] += 1
                self.stats["total_digests"] += 1
                approved_roles.add(item["agent"])
                self.stats["gate_decisions"][gate_decision_idx]["digest"] = True
            else:
                blocked_items.append((item, gate_decision_idx))

        blocked_items_with_idx = blocked_items
        blocked_items_plain = [x[0] for x in blocked_items_with_idx]
        rescued = self._rescue_required_roles(
            state=state,
            blocked_items=blocked_items_plain,
            approved_broadcasts=approved_broadcasts,
            running_history=running_history,
            approved_roles=approved_roles
        )
        # _rescue_required_roles currently returns remaining blocked items.
        # Mark rescued candidates: anything originally blocked *not* in remaining rescued list.
        rescued_set_ids = set()
        if len(rescued) != len(blocked_items_plain):
            # Approximate: if lengths differ, assume first (N-diff) blocked items were rescued.
            for (_, idx) in blocked_items_with_idx[: len(blocked_items_with_idx) - len(rescued)]:
                self.stats["gate_decisions"][idx]["rescued"] = True
                self.stats["gate_decisions"][idx]["is_allowed"] = True
        self.stats["total_blocked"] += len(rescued)
        blocked_items = rescued
        
        self.stats["history_per_step"].append({
            "step": state["step_t"],
            "candidates": len(candidates),
            "approved": len(approved_broadcasts),
            "blocked": len(candidates) - len(approved_broadcasts),
            "digests": self.stats["total_digests"],
            "role_rescues": self.stats["total_role_rescues"]
        })
        
        return {
            "history": approved_broadcasts,
            "candidates": [], 
            "step_t": state["step_t"] + 1,
            "total_candidates": state.get("total_candidates", 0) + len(candidates),
            "total_approved": state.get("total_approved", 0) + len(approved_broadcasts)
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue or end the workflow."""
        self.stats["steps_completed"] = state["step_t"] - 1
        
        # Check completion
        if self.completion_checker(state["history"], state["global_goal"]):
            return "end"
        
        # 修改点：当 step_t 严格大于 max_steps 时才退出，确保第 2 轮能够正常执行
        if state["step_t"] > self.max_steps:
            return "end"
        
        return "continue"
    
    def run(self, goal: str, verbose: bool = True) -> Dict:
        """
        Run the multi-agent workflow.
        
        Args:
            goal: The global task goal
            verbose: Whether to print progress
            
        Returns:
            Final state and statistics
        """
        # Reset statistics
        self.stats = {
            "total_candidates": 0,
            "total_approved": 0,
            "total_blocked": 0,
            "total_digests": 0,
            "total_role_rescues": 0,
            "steps_completed": 0,
            "history_per_step": [],
            "gate_decisions": []
        }
        
        # Initial state
        initial_state = {
            "global_goal": goal,
            "history": [],
            "candidates": [],
            "step_t": 1,
            "total_candidates": 0,
            "total_approved": 0
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting VIB Multi-Agent Workflow")
            print(f"Goal: {goal}")
            print(f"Agents: {[a['name'] for a in self.agents]}")
            print(f"Max Steps: {self.max_steps}")
            print(f"{'='*60}\n")
        
        # Run the workflow and accumulate history from gatekeeper outputs
        final_state = None
        accumulated_history = []
        for step_output in self.app.stream(initial_state):
            for node_name, output in step_output.items():
                if node_name == "vib_gatekeeper":
                    # Accumulate approved messages from each step
                    accumulated_history.extend(output.get("history", []))
                    if verbose:
                        step_stats = self.stats["history_per_step"][-1]
                        print(f"Step {step_stats['step']}: "
                              f"{step_stats['approved']}/{step_stats['candidates']} approved, "
                              f"{step_stats['blocked']} blocked")
            final_state = step_output
        
        # Calculate compression rate
        total = self.stats["total_candidates"]
        approved = self.stats["total_approved"]
        compression_rate = (1 - approved / total) * 100 if total > 0 else 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Workflow Complete!")
            print(f"Steps: {self.stats['steps_completed']}")
            print(f"Total Candidates: {total}")
            print(f"Approved: {approved}")
            print(f"Blocked: {self.stats['total_blocked']}")
            print(f"Digests: {self.stats['total_digests']}")
            print(f"Compression Rate: {compression_rate:.1f}%")
            print(f"{'='*60}\n")
        
        return {
            "final_state": final_state,
            "history": accumulated_history,
            "stats": self.stats,
            "compression_rate": compression_rate
        }


# ============================================
# Mock Agents for Testing (Milestone 2)
# ============================================

def create_mock_programmer_agent():
    """Create a mock programmer agent for testing."""
    responses = [
        "I suggest we use quicksort for this optimization task. It has O(n log n) average complexity.",
        "Looking at the code, we should also consider the cache locality of our algorithm.",
        "For nearly sorted data, insertion sort might actually be faster as a hybrid approach.",
        "We should add benchmarking code to measure the actual performance improvement.",
        "The implementation looks good. Let's add some edge case handling.",
        "I recommend we also consider memory usage in our optimization.",
        "The code is well-structured. Perhaps we can add parallel processing?",
        "Final review: the solution handles all the requirements efficiently."
    ]
    call_count = [0]
    
    def generate(goal: str, history: List[str], step_t: int) -> str:
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]
    
    return {"name": "Programmer", "generate": generate}


def create_mock_reviewer_agent():
    """Create a mock reviewer agent for testing."""
    responses = [
        "I agree with using quicksort. The O(n log n) complexity is optimal for general sorting.",
        "Good point about cache locality. We should also consider branch prediction.",
        "The hybrid approach with insertion sort is a well-known optimization technique.",
        "Benchmarking is essential. We should test with various input sizes and distributions.",
        "Edge cases like empty arrays and single elements should be handled explicitly.",
        "Memory optimization is important. In-place sorting would be preferable.",
        "Parallel processing could help for very large datasets.",
        "The solution is complete and well-tested."
    ]
    call_count = [0]
    
    def generate(goal: str, history: List[str], step_t: int) -> str:
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]
    
    return {"name": "Reviewer", "generate": generate}


def create_mock_redundant_agent():
    """Create a mock agent that produces redundant messages (for testing blocking)."""
    def generate(goal: str, history: List[str], step_t: int) -> str:
        # Always produce similar/redundant content
        return "Yes, I completely agree with the previous suggestions. This is a great approach."
    
    return {"name": "Redundant", "generate": generate}


def run_milestone2_demo():
    """
    Run Milestone 2 demonstration with mock agents.
    """
    print("\n" + "="*60)
    print("Milestone 2: LangGraph State Machine Demo")
    print("="*60)
    
    # Create agents
    agents = [
        create_mock_programmer_agent(),
        create_mock_reviewer_agent(),
        create_mock_redundant_agent()  # This one should get blocked often
    ]
    
    # Create VIB router
    vib_router = VIBSparseCommRouter(
        beta_0=2.0,
        lambda_decay=0.15,
        temp_tau=0.2,
        gamma=0.0
    )
    
    # Create and run workflow
    workflow = VIBMultiAgentWorkflow(
        agents=agents,
        vib_router=vib_router,
        max_steps=5
    )
    
    result = workflow.run(
        goal="Optimize the sorting performance of this Python code",
        verbose=True
    )
    
    return result


if __name__ == "__main__":
    run_milestone2_demo()

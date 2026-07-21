"""
LangGraph State Definition for VIB-SparseComm Multi-Agent System
"""

import operator
from typing import Annotated, List, Dict, TypedDict


class AgentState(TypedDict):
    """
    Global shared state for the multi-agent collaboration system.
    
    Attributes:
        global_goal: The global task objective/instruction
        history: Public communication history (approved messages only, readable by all agents)
        candidates: Temporary buffer for pending candidate messages from agents
        step_t: Current collaboration round number
    """
    global_goal: str                              # Global task objective
    history: Annotated[List[str], operator.add]   # Public history (accumulated via reducer)
    candidates: List[Dict[str, str]]              # Temporary buffer for candidate messages
    step_t: int                                   # Current collaboration step t

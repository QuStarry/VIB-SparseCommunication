"""
VIB-SparseCommunication: Variational Information Bottleneck based Sparse Communication
for Multi-Agent Systems
"""

from .vib_router import VIBSparseCommRouter, VIBEvaluationResult
from .state import AgentState
from .workflow import VIBMultiAgentWorkflow

__all__ = [
    "VIBSparseCommRouter",
    "VIBEvaluationResult", 
    "AgentState",
    "VIBMultiAgentWorkflow"
]

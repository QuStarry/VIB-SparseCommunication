"""
Variational Information Bottleneck (VIB) Sparse Communication Router

This module implements the core VIB-based message filtering mechanism that determines
whether an agent's candidate message should be broadcast to the shared history.

Mathematical Foundation:
    - Optimization Objective: min L_VIB(M) = I(M; H^{t-1}) - β_t * I(M; G)
    - Sufficiency: I(M; G) ≈ Sim(E(M), E(G))
    - Redundancy: I(M; H) ≈ log Σ exp(Sim(E(M), E(h)) / τ)
    - Dynamic decay: β_t = β_0 * exp(-λ * t)
    - Gate decision: Utility(M) = β_t * I(M; G) - I(M; H) >= γ
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VIBEvaluationResult:
    """Result of VIB evaluation with detailed metrics."""
    utility_score: float          # Final utility score
    sufficiency: float            # I(M; G) - task relevance
    redundancy: float             # I(M; H) - history redundancy
    beta_t: float                 # Dynamic Lagrange multiplier at step t
    is_allowed: bool              # Whether the message is allowed to broadcast
    threshold: float              # The gamma threshold used


class VIBSparseCommRouter:
    """
    Variational Information Bottleneck Router for Sparse Multi-Agent Communication.
    
    This router evaluates candidate messages from agents and determines whether they
    contain sufficient novel, task-relevant information to justify broadcasting to
    the shared communication history.
    
    Args:
        beta_0: Initial Lagrange multiplier (controls early exploration). Default: 2.0
        lambda_decay: Exponential decay rate for β_t. Default: 0.15
        temp_tau: Temperature parameter for contrastive density estimation. Default: 0.2
        gamma: Base admission threshold. Default: 0.0
        embedding_model: Name of the sentence-transformers model. Default: "all-MiniLM-L6-v2"
    """
    
    def __init__(
        self,
        beta_0: float = 2.0,
        lambda_decay: float = 0.15,
        temp_tau: float = 0.2,
        gamma: float = 0.0,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.beta_0 = beta_0
        self.lambda_decay = lambda_decay
        self.tau = temp_tau
        self.gamma = gamma
        self.model_name = embedding_model
        
        # Lazy load the embedding model
        self._encoder = None
        
        # Cache for embeddings to avoid redundant computation
        self._embedding_cache: dict = {}
    
    @property
    def encoder(self):
        """Lazy load the sentence-transformers model."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
        return self._encoder
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get L2-normalized embedding vector for the given text.
        
        Args:
            text: Input text to encode
            
        Returns:
            L2-normalized embedding vector as numpy array
        """
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # Encode and L2-normalize
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Cache the result
        self._embedding_cache[text] = embedding
        return embedding
    
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Since vectors are L2-normalized, this is equivalent to dot product.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return float(dot_product / (norm_a * norm_b + 1e-9))
    
    def compute_sufficiency(self, msg_embedding: np.ndarray, goal_embedding: np.ndarray) -> float:
        """
        Compute task sufficiency I(M; G) ≈ Sim(E(M), E(G)).
        
        This measures how relevant the candidate message is to the global task goal.
        Higher values indicate the message contains information useful for achieving the goal.
        
        Args:
            msg_embedding: Embedding of candidate message M
            goal_embedding: Embedding of global goal G
            
        Returns:
            Sufficiency score (cosine similarity)
        """
        return self._cosine_similarity(msg_embedding, goal_embedding)
    
    def compute_redundancy(
        self, 
        msg_embedding: np.ndarray, 
        history_embeddings: List[np.ndarray]
    ) -> float:
        """
        Compute history redundancy I(M; H^{t-1}) using contrastive density estimation.
        
        Based on InfoNCE-style estimation:
        I(M; H) ≈ log Σ_{h ∈ H} exp(Sim(E(M), E(h)) / τ)
        
        Higher values indicate the message is redundant with existing history.
        
        Args:
            msg_embedding: Embedding of candidate message M
            history_embeddings: List of embeddings for history messages
            
        Returns:
            Redundancy score (log-sum-exp of similarities)
        """
        if not history_embeddings:
            return 0.0
        
        # Compute similarities with all history items
        sim_scores = [
            self._cosine_similarity(msg_embedding, h_emb) 
            for h_emb in history_embeddings
        ]
        
        # Log-Sum-Exp for numerical stability
        # I(M; H) ≈ log Σ exp(sim / τ)
        scaled_sims = [s / self.tau for s in sim_scores]
        max_sim = max(scaled_sims)
        
        # Numerically stable log-sum-exp
        log_sum_exp = max_sim + math.log(
            sum(math.exp(s - max_sim) for s in scaled_sims) + 1e-9
        )
        
        return log_sum_exp
    
    def compute_beta_t(self, step_t: int) -> float:
        """
        Compute the dynamic Lagrange multiplier β_t.
        
        β_t = β_0 * exp(-λ * t)
        
        Early rounds (small t): β_t is large → encourages exploration
        Later rounds (large t): β_t decays → enforces compression
        
        Args:
            step_t: Current collaboration step
            
        Returns:
            Dynamic β_t value
        """
        return self.beta_0 * math.exp(-self.lambda_decay * step_t)
    
    def evaluate(
        self, 
        candidate_msg: str, 
        history: List[str], 
        goal: str, 
        step_t: int
    ) -> bool:
        """
        Core evaluation: determine whether to allow message broadcast.
        
        Args:
            candidate_msg: The candidate message from an agent
            history: Current public communication history
            goal: Global task goal
            step_t: Current collaboration step
            
        Returns:
            True if message should be broadcast, False if it should be pruned
        """
        result = self.evaluate_detailed(candidate_msg, history, goal, step_t)
        return result.is_allowed
    
    def evaluate_detailed(
        self, 
        candidate_msg: str, 
        history: List[str], 
        goal: str, 
        step_t: int
    ) -> VIBEvaluationResult:
        """
        Detailed evaluation with full metrics for analysis and debugging.
        
        Args:
            candidate_msg: The candidate message from an agent
            history: Current public communication history
            goal: Global task goal
            step_t: Current collaboration step
            
        Returns:
            VIBEvaluationResult with all computed metrics
        """
        # First round: unconditionally allow to establish initial context
        if not history:
            return VIBEvaluationResult(
                utility_score=float('inf'),
                sufficiency=1.0,
                redundancy=0.0,
                beta_t=self.compute_beta_t(step_t),
                is_allowed=True,
                threshold=self.gamma
            )
        
        # Get embeddings
        emb_m = self.get_embedding(candidate_msg)
        emb_g = self.get_embedding(goal)
        emb_history = [self.get_embedding(h) for h in history]
        
        # Compute sufficiency I(M; G)
        sufficiency = self.compute_sufficiency(emb_m, emb_g)
        
        # Compute redundancy I(M; H)
        redundancy = self.compute_redundancy(emb_m, emb_history)
        
        # Compute dynamic β_t
        beta_t = self.compute_beta_t(step_t)
        
        # Compute final utility score
        # Utility(M) = β_t * I(M; G) - I(M; H)
        utility_score = beta_t * sufficiency - redundancy
        
        # Gate decision
        is_allowed = utility_score >= self.gamma
        
        return VIBEvaluationResult(
            utility_score=utility_score,
            sufficiency=sufficiency,
            redundancy=redundancy,
            beta_t=beta_t,
            is_allowed=is_allowed,
            threshold=self.gamma
        )
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
    
    def batch_evaluate(
        self,
        candidates: List[dict],
        history: List[str],
        goal: str,
        step_t: int
    ) -> List[Tuple[dict, VIBEvaluationResult]]:
        """
        Batch evaluate multiple candidate messages.
        
        Args:
            candidates: List of candidate dicts with 'agent' and 'content' keys
            history: Current public communication history
            goal: Global task goal
            step_t: Current collaboration step
            
        Returns:
            List of (candidate, result) tuples
        """
        results = []
        for candidate in candidates:
            result = self.evaluate_detailed(
                candidate_msg=candidate["content"],
                history=history,
                goal=goal,
                step_t=step_t
            )
            results.append((candidate, result))
        return results

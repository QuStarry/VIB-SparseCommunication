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

Ada-VIB (Adaptive Uncertainty-Guided VIB):
    - Dynamic threshold: τ_adaptive = max(τ_min, τ_0 - α * H̄(Y))
    - H̄(Y): Normalized message entropy / uncertainty proxy
    - When agent output has high uncertainty (high entropy), threshold is lowered
      to allow more detailed intermediate reasoning to pass through
    - When confidence is high (low entropy), threshold stays high for strong pruning
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ============================================
# Ada-VIB: Adaptive Threshold Utility Functions
# ============================================

def compute_message_entropy(token_probs_or_logits: Optional[List[float]] = None) -> float:
    """
    计算生成消息的平均 Token 熵，作为不确定度指标。
    
    当有 API 返回的 logprobs 时，直接计算 Shannon Entropy：
        H = -Σ p(x) * log2(p(x))
    
    Args:
        token_probs_or_logits: 每个 token 的概率列表 (0~1)
        
    Returns:
        归一化平均熵值 (float)，无数据时返回 0.0
    """
    if not token_probs_or_logits:
        return 0.0
    
    # 计算均值熵 (Normalized Entropy)
    entropies = [-p * math.log2(p) for p in token_probs_or_logits if p > 0]
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    return mean_entropy


def get_adaptive_threshold(
    base_tau: float = 0.5,
    alpha: float = 0.2,
    tau_min: float = 0.25,
    token_probs: Optional[List[float]] = None,
    confidence_score: Optional[float] = None
) -> float:
    """
    自适应计算当前消息的 VIB 剪枝阈值 (Ada-VIB)。
    
    公式：τ_adaptive = max(τ_min, τ_0 - α * H̄(Y))
    
    支持两种不确定度来源：
    1. Token Logprobs → 计算 Shannon Entropy
    2. Confidence Score (0~1) → 使用 (1 - confidence) 作为不确定度
    
    熵越高 / 置信度越低 → 阈值越低 → 消息越容易被保留传输
    
    Args:
        base_tau: 基础门控阈值 τ_0
        alpha: 自适应调节系数 (敏感度超参数)
        tau_min: 保底门控下限，防止退化回 Full Comm
        token_probs: Token 概率列表 (可选)
        confidence_score: Agent 自评置信度 0~1 (可选，轻量替代)
        
    Returns:
        动态调整后的阈值 τ_adaptive
    """
    if confidence_score is not None:
        # 轻量化替代方案：τ_adaptive = τ_0 - α * (1 - confidence)
        uncertainty = 1.0 - confidence_score
    else:
        # 基于 Token 熵的标准方案
        uncertainty = compute_message_entropy(token_probs)
    
    # 动态调低阈值：熵越高，阈值越低，越容易被保留并传输
    adaptive_tau = base_tau - (alpha * uncertainty)
    return max(tau_min, adaptive_tau)


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
    
    Supports Ada-VIB (Adaptive Uncertainty-Guided VIB) mode:
        When adaptive=True, the gate threshold γ is dynamically adjusted based on
        the estimated uncertainty (entropy) of the candidate message:
        τ_adaptive = max(τ_min, γ - α * H̄(Y))
    
    Args:
        beta_0: Initial Lagrange multiplier (controls early exploration). Default: 2.0
        lambda_decay: Exponential decay rate for β_t. Default: 0.15
        temp_tau: Temperature parameter for contrastive density estimation. Default: 0.2
        gamma: Base admission threshold. Default: 0.0
        embedding_model: Name of the sentence-transformers model. Default: "all-MiniLM-L6-v2"
        adaptive: Enable Ada-VIB adaptive threshold mode. Default: False
        alpha: Adaptive sensitivity coefficient (how strongly entropy affects threshold). Default: 0.2
        tau_min: Minimum threshold floor to prevent degeneration to Full Comm. Default: -0.5
    """
    
    def __init__(
        self,
        beta_0: float = 2.0,
        lambda_decay: float = 0.15,
        temp_tau: float = 0.2,
        gamma: float = 0.0,
        embedding_model: str = "all-MiniLM-L6-v2",
        adaptive: bool = False,
        alpha: float = 0.2,
        tau_min: float = -0.5
    ):
        self.beta_0 = beta_0
        self.lambda_decay = lambda_decay
        self.tau = temp_tau
        self.gamma = gamma
        self.model_name = embedding_model
        
        # Ada-VIB: Adaptive threshold parameters
        self.adaptive = adaptive
        self.alpha = alpha
        self.tau_min = tau_min
        
        # Lazy load the embedding model
        self._encoder = None
        
        # Cache for embeddings to avoid redundant computation
        self._embedding_cache: dict = {}
    
    @property
    def encoder(self):
        """Lazy load the sentence-transformers model (offline-first to avoid SSL timeouts)."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                # 优先尝试离线加载（模型已缓存时避免联网校验）
                try:
                    self._encoder = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
                    # 本地无缓存时回退到在线下载
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
            if not history_embeddings:
                return 0.0
            
            sim_scores = [
                self._cosine_similarity(msg_embedding, h_emb) 
                for h_emb in history_embeddings
            ]
            
            # 使用温度系数 τ 进行缩放
            scaled_sims = [s / self.tau for s in sim_scores]
            max_sim = max(scaled_sims)
            
            # 数值稳定的 Log-Sum-Exp 计算
            log_sum_exp = max_sim + math.log(
                sum(math.exp(s - max_sim) for s in scaled_sims) + 1e-9
            )
            
            # 【优化点】：对历史消息数量进行平滑开方缩放，避免冗余度随历史条数无限制暴胀
            history_factor = math.sqrt(len(history_embeddings))
            return log_sum_exp / history_factor
    
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
    
    def compute_entropy_proxy(
        self,
        msg_embedding: np.ndarray,
        history_embeddings: List[np.ndarray],
        candidate_msg: str
    ) -> float:
        """
        计算消息的文本级不确定度代理 (Entropy Proxy)，用于 Ada-VIB 自适应门控。
        
        当无法获取 Token Logprobs 时，使用以下两个信号估计消息不确定度：
        1. 语义新颖度 (Semantic Novelty): 消息与历史的最大相似度越低 → 不确定度越高
        2. 词汇多样性 (Lexical Diversity): 消息中独特词汇比例越高 → 信息密度越大
        
        归一化输出范围 [0, 1]：
        - 0.0 = 完全冗余/确定性极高（简单附和）
        - 1.0 = 完全新颖/不确定性极高（复杂中间推导）
        
        Args:
            msg_embedding: 候选消息的嵌入向量
            history_embeddings: 历史消息的嵌入向量列表
            candidate_msg: 候选消息原文
            
        Returns:
            归一化不确定度代理值 [0, 1]
        """
        # --- 信号 1: 语义新颖度 (Semantic Novelty) ---
        if history_embeddings:
            similarities = [
                self._cosine_similarity(msg_embedding, h_emb)
                for h_emb in history_embeddings
            ]
            max_sim = max(similarities)
            # 新颖度 = 1 - 最大相似度 (sim 越高越冗余，novelty 越低)
            # 裁剪到 [0, 1] 范围 (cosine sim 可能为负)
            novelty = max(0.0, min(1.0, 1.0 - max_sim))
        else:
            # 无历史时，首条消息默认中等不确定度
            novelty = 0.5
        
        # --- 信号 2: 词汇多样性 (Lexical Diversity / Type-Token Ratio) ---
        words = candidate_msg.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
        else:
            unique_ratio = 0.0
        
        # --- 综合不确定度: 加权平均 (语义新颖度权重 0.7, 词汇多样性权重 0.3) ---
        entropy_proxy = 0.7 * novelty + 0.3 * unique_ratio
        
        return float(np.clip(entropy_proxy, 0.0, 1.0))
    
    def compute_adaptive_gamma(
        self,
        msg_embedding: np.ndarray,
        history_embeddings: List[np.ndarray],
        candidate_msg: str,
        token_probs: Optional[List[float]] = None,
        confidence_score: Optional[float] = None
    ) -> float:
        """
        计算 Ada-VIB 自适应动态门控阈值。
        
        核心公式：τ_adaptive = max(τ_min, γ - α * H̄(Y))
        
        优先使用 Token Logprobs / Confidence Score；
        若均不可用，则使用文本级 Entropy Proxy 作为不确定度估计。
        
        Args:
            msg_embedding: 候选消息嵌入
            history_embeddings: 历史消息嵌入列表
            candidate_msg: 候选消息原文
            token_probs: Token 概率列表 (可选，来自 API logprobs)
            confidence_score: Agent 自评置信度 (可选)
            
        Returns:
            动态调整后的准入阈值 γ_adaptive
        """
        if confidence_score is not None:
            # 方案 A: 基于 Agent 自评置信度
            uncertainty = 1.0 - confidence_score
        elif token_probs:
            # 方案 B: 基于 Token Logprobs 的 Shannon Entropy
            uncertainty = compute_message_entropy(token_probs)
        else:
            # 方案 C (默认): 基于文本级 Entropy Proxy
            uncertainty = self.compute_entropy_proxy(
                msg_embedding, history_embeddings, candidate_msg
            )
        
        # 动态阈值: 不确定度越高 → 阈值越低 → 消息越容易通过
        adaptive_gamma = self.gamma - (self.alpha * uncertainty)
        return max(self.tau_min, adaptive_gamma)
    
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
        step_t: int,
        token_probs: Optional[List[float]] = None,
        confidence_score: Optional[float] = None
    ) -> VIBEvaluationResult:
        """
        Detailed evaluation with full metrics for analysis and debugging.
        
        Supports Ada-VIB mode: when self.adaptive=True, the threshold γ is
        dynamically adjusted based on message uncertainty (entropy proxy).
        
        Args:
            candidate_msg: The candidate message from an agent
            history: Current public communication history
            goal: Global task goal
            step_t: Current collaboration step
            token_probs: Optional token probabilities for entropy calculation
            confidence_score: Optional agent self-assessed confidence (0~1)
            
        Returns:
            VIBEvaluationResult with all computed metrics
        """
        # 删掉原来的 if not history: return True 全放行逻辑
        # 改为：仅当 history 为空且当前是第一个候选者时，计算 base utility，避免首条就被剔除，但允许后续消息被精准剪枝。

        if not history:
            # 只有当全局+增量历史完全为空时，计算与 Goal 的相关度
            emb_m = self.get_embedding(candidate_msg)
            emb_g = self.get_embedding(goal)
            sufficiency = self.compute_sufficiency(emb_m, emb_g)
            beta_t = self.compute_beta_t(step_t)
            utility_score = beta_t * sufficiency  # 此时 redundancy 为 0
            
            # 依然走动态阈值判断，防止极低质量/幻觉首条消息混入
            effective_threshold = self.compute_adaptive_gamma(
                msg_embedding=emb_m, history_embeddings=[], candidate_msg=candidate_msg,
                token_probs=token_probs, confidence_score=confidence_score
            ) if self.adaptive else self.gamma
            
            return VIBEvaluationResult(
                utility_score=utility_score,
                sufficiency=sufficiency,
                redundancy=0.0,
                beta_t=beta_t,
                is_allowed=(utility_score >= effective_threshold),
                threshold=effective_threshold
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
        
        # Ada-VIB: Compute adaptive threshold or use fixed gamma
        if self.adaptive:
            effective_threshold = self.compute_adaptive_gamma(
                msg_embedding=emb_m,
                history_embeddings=emb_history,
                candidate_msg=candidate_msg,
                token_probs=token_probs,
                confidence_score=confidence_score
            )
        else:
            effective_threshold = self.gamma
        
        # Gate decision
        is_allowed = utility_score >= effective_threshold
        
        return VIBEvaluationResult(
            utility_score=utility_score,
            sufficiency=sufficiency,
            redundancy=redundancy,
            beta_t=beta_t,
            is_allowed=is_allowed,
            threshold=effective_threshold
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

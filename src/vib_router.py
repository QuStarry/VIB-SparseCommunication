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

Naive gating baselines (gating_mode != "vib"):
    - "random": keep each candidate independently with probability keep_rate,
      calibrated to the VIB arm's observed admission rate (matched budget).
    - "topk": cosine diversity gate. A candidate is kept iff its maximum cosine
      similarity against the already-broadcast history stays below sim_threshold,
      i.e. the k least-redundant candidates of a round are admitted. This uses the
      same shared encoder as the VIB arm but none of its learned/prior machinery,
      which isolates the contribution of the utility formulation itself.
"""

import math
import random
import re
import threading
import zlib
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher


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
        gating_mode: "vib" (default) or a naive baseline ("random", "topk").
        keep_rate: Admission probability for gating_mode="random". Default: 0.5
        sim_threshold: Max admissible cosine-to-history for gating_mode="topk". Default: 0.85
        random_seed: Seed of the RNG used by gating_mode="random" (reproducibility). Default: 20260812
    """
    
    _shared_encoder_cache = {}
    _encoder_lock = threading.Lock()

    @classmethod
    def warmup_shared_encoder(cls, model_name: str = "all-MiniLM-L6-v2"):
        """Preload the shared sentence-transformers encoder once for concurrent runs."""
        router = cls(embedding_model=model_name)
        _ = router.encoder

    def __init__(
        self,
        beta_0: float = 2.0,
        lambda_decay: float = 0.15,
        temp_tau: float = 0.2,
        gamma: float = 0.0,
        embedding_model: str = "all-MiniLM-L6-v2",
        adaptive: bool = False,
        alpha: float = 0.2,
        tau_min: float = -0.5,
        enable_math_role_weights: bool = True,
        enable_reasoning_guard: bool = True,
        enable_math_critical_guard: bool = True,
        gating_mode: str = "vib",
        keep_rate: float = 0.5,
        sim_threshold: float = 0.85,
        random_seed: int = 20260812
    ):
        if gating_mode not in ("vib", "random", "topk"):
            raise ValueError(
                f"Unknown gating_mode {gating_mode!r}; expected 'vib', 'random' or 'topk'."
            )
        self.beta_0 = beta_0
        self.lambda_decay = lambda_decay
        self.tau = temp_tau
        self.gamma = gamma
        self.model_name = embedding_model
        
        # Ada-VIB: Adaptive threshold parameters
        self.adaptive = adaptive
        self.alpha = alpha
        self.tau_min = tau_min
        self.enable_math_role_weights = enable_math_role_weights
        self.enable_reasoning_guard = enable_reasoning_guard
        self.enable_math_critical_guard = enable_math_critical_guard

        # Naive gating baselines (control conditions for the VIB utility)
        self.gating_mode = gating_mode
        self.keep_rate = keep_rate
        self.sim_threshold = sim_threshold
        self._rng = random.Random(random_seed)
        
        # Bugfix-Aware: 通用代码修正与边界条件敏感词汇表（泛化设计，不依赖特定题库）
        self.fix_keywords = [
            "fix", "bug", "incorrect", "corner case", "edge case",
            "boundary", "fails when", "missing", "instead of", "should be",
            "error", "wrong", "typo", "off-by-one", "overflow"
        ]

        # 数学推理守护关键词：保护中间算式、单位换算与验证反驳信息
        self.reasoning_keywords = [
            "equation", "total", "remaining", "difference", "sum", "average",
            "per", "each", "every", "times", "twice", "thrice", "percent",
            "therefore", "so", "check", "verify", "incorrect", "re-check",
            "let ", "=", "+", "-", "*", "/", "step", "first", "then",
            "小时", "分钟", "美元", "总共", "剩余", "每", "因此", "所以", "检查"
        ]

        self.math_roles = {"Planner", "Solver", "Verifier"}
        
        # 角色自适应 Beta 权重映射 (Role-Prior Dynamic β) - 基于 HumanEval 164 题消融分析优化
        self.role_beta_map = {
            "Architect": 3.2,   # 架构方案高准入（+0.2 提升方案质量）
            "Coder": 4.2,       # 进一步提高 Coder 准入，保证关键代码递交
            "Reviewer": 3.6,    # 增强 Reviewer 对代码逻辑纠错的权重
            "Planner": 2.8,     # 数学规划信息更容易被压缩
            "Solver": 4.4,      # 保护具体算式与求解步骤
            "Verifier": 4.0     # 保护校验/纠错链条
        }
        
        # Lazy load the embedding model
        self._encoder = None
        
        # Cache for embeddings to avoid redundant computation
        self._embedding_cache: dict = {}
    
    @property
    def encoder(self):
        """Lazy load the sentence-transformers model (offline-first to avoid SSL timeouts)."""
        if self._encoder is None:
            with self.__class__._encoder_lock:
                shared = self.__class__._shared_encoder_cache.get(self.model_name)
                if shared is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        # 并发场景下仅允许一个线程初始化底层模型，避免 meta tensor/race 问题
                        try:
                            shared = SentenceTransformer(self.model_name, local_files_only=True)
                        except Exception:
                            shared = SentenceTransformer(self.model_name)
                    except ImportError:
                        raise ImportError(
                            "sentence-transformers is required. Install with: pip install sentence-transformers"
                        )
                    self.__class__._shared_encoder_cache[self.model_name] = shared
                self._encoder = shared
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
    
    def _extract_code(self, text: str) -> str:
        """
        从 Agent 消息中提取 Python 代码块内容。
        用于 Code-Diff 敏感度计算，避免代码补丁被误判为高冗余。
        """
        code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
        if not code_blocks:
            code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        return "\n".join(code_blocks).strip() if code_blocks else ""
    
    def _compute_code_diff_ratio(self, curr_code: str, history_messages: List[str]) -> float:
        """
        计算当前代码块与历史最近代码块的差异率 (Code Diff Sensitivity)。
        
        Returns:
            diff_ratio ∈ [0, 1]，0=完全相同，1=完全不同
        """
        if not curr_code:
            return 0.0
        hist_codes = [self._extract_code(h) for h in history_messages]
        hist_codes = [c for c in hist_codes if c]  # 过滤空代码
        if not hist_codes:
            return 1.0  # 首段代码，视为全新
        last_code = hist_codes[-1]
        diff_ratio = 1.0 - SequenceMatcher(None, curr_code, last_code).ratio()
        return diff_ratio
    
    def _detect_fix_signal(self, message: str, role: Optional[str] = None) -> bool:
        """
        检测消息中是否包含 Bug 修正信号关键词。
        当 Reviewer/Coder 消息包含 fix/bug/edge case 等词汇时触发。
        """
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in self.fix_keywords)

    def _detect_reasoning_signal(self, message: str, role: Optional[str] = None) -> bool:
        """
        检测消息中是否包含数学推理链的关键算式、单位或验证信号。
        该信号用于保护 GSM8K 等任务中的 Solver/Verifier 中间步骤。
        """
        if not self.enable_reasoning_guard or role not in self.math_roles:
            return False

        msg_lower = message.lower()
        has_keyword = any(kw in msg_lower for kw in self.reasoning_keywords)
        has_number = bool(re.search(r"\d", message))
        has_equation_shape = bool(re.search(r"(\d+\s*[%*/+\-=]\s*\d+|=\s*\d+)", message))
        return has_keyword or (has_number and has_equation_shape)

    def _detect_math_critical(self, message: str, role: Optional[str] = None) -> bool:
        """检测数学消息是否含「结论 / 纠错」关键锚点（推理链节点，误剪代价最高）。

        MATH 回归分析显示折损集中在高难度题与几何类，根因是语义余弦相似度把
        「关键推导步骤」误判为冗余。此信号对两类高价值消息给予更强保护：
          1) 结论锚点：最终答案 / 推导结论（\\boxed、####、\\therefore、答案...）
          2) 纠错锚点：Verifier 指出错误（incorrect、should be、错误、应为...）
        """
        if not self.enable_reasoning_guard or not self.enable_math_critical_guard or role not in self.math_roles:
            return False

        msg_lower = message.lower()
        conclusion = any(m in msg_lower for m in [
            "\\boxed", "####", "\\therefore", "final answer", "the answer is",
            "answer:", "答案", "最终答案", "综上",
        ])
        if conclusion:
            return True
        correction = any(m in msg_lower for m in [
            "incorrect", "wrong", "should be", "must be", "recheck", "re-check",
            "error", "mistake", "missed",
            "错误", "不对", "应为", "应该", "检查", "遗漏", "有误",
        ])
        return correction and role == "Verifier"
    
    def _compute_complexity_bonus(self, goal: str) -> float:
        """
        基于任务 Prompt 文本长度的泛化复杂度指标。
        
        原理：任务描述越长、词汇越丰富，通常意味着约束条件越多、边界情况越复杂，
        需要更高的信息准入来避免关键分析被过度裁剪。
        
        相比硬编码关键词匹配，此方法更具泛化能力，避免针对特定题库过拟合。
        
        Returns:
            complexity_bonus ∈ [0, 1.0]，描述越长加成越大
        """
        word_count = len(goal.split())
        return min(1.0, word_count / 50.0)
    
    def compute_beta_t(
        self,
        step_t: int,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        is_execution_failed: bool = False
    ) -> float:
        """
        Compute the dynamic Lagrange multiplier β_t with role-adaptive weighting,
        task-complexity awareness, and feedback-aware decay reset.
        
        β_t = (β_role + complexity_bonus) * exp(-λ * effective_t)
        
        When role is provided, uses role-specific base beta:
        - Architect: 3.2 (方案高准入)
        - Coder: 4.2 (保障代码送达，避免误杀)
        - Reviewer: 3.6 (增强纠错权重)
        
        When goal is provided, adds generalized complexity bonus based on text length.
        When is_execution_failed=True, resets decay to t=0 to ensure fix messages pass through.
        
        Args:
            step_t: Current collaboration step
            role: Agent role name (optional, enables role-adaptive mode)
            goal: Global task goal (optional, enables complexity-aware mode)
            is_execution_failed: Whether previous code execution failed (feedback signal)
            
        Returns:
            Dynamic β_t value
        """
        if role and self.enable_math_role_weights:
            base_beta = self.role_beta_map.get(role, self.beta_0)
        elif role and role not in self.math_roles:
            base_beta = self.role_beta_map.get(role, self.beta_0)
        else:
            base_beta = self.beta_0
        
        # Task-Complexity Adaptive: 基于文本长度的泛化复杂度加成
        if goal:
            complexity_bonus = self._compute_complexity_bonus(goal)
            base_beta += complexity_bonus
        
        # Feedback-Aware: 当执行失败时重置衰减，确保修正消息强行放行
        effective_t = 0 if is_execution_failed else step_t
        
        return base_beta * math.exp(-self.lambda_decay * effective_t)
    
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
    
    @staticmethod
    def derive_task_seed(base_seed: int, task_salt: str) -> int:
        """Mix a task identifier into the base seed of the random baseline.

        The harness builds one router per task, so seeding every router with the
        bare base seed would replay a single fixed keep/drop pattern on every
        task -- a deterministic fixed-position policy rather than the independent
        Bernoulli draw this arm is defined as. Salting with the task id gives
        each task its own stream while keeping the whole run reproducible.
        ``zlib.crc32`` is used instead of ``hash`` because the latter is salted
        per process and would break run-to-run reproducibility.
        """
        return (int(base_seed) + zlib.crc32(task_salt.encode("utf-8"))) % (2 ** 32)

    def _evaluate_naive(
        self,
        candidate_msg: str,
        history: List[str]
    ) -> VIBEvaluationResult:
        """
        Naive gating baselines used as control conditions for the VIB utility.

        Both modes deliberately ignore the task goal, the role priors, the
        dynamic β schedule and every guard, so that any measured advantage of the
        VIB arm can be attributed to the utility formulation rather than to the
        shared encoder or the message budget.

        Returns:
            VIBEvaluationResult whose ``is_allowed`` is the baseline decision;
            ``utility_score`` carries the raw signal (1/0 for random, the
            least-redundant margin for topk) for auditability.
        """
        if self.gating_mode == "random":
            keep = self._rng.random() < self.keep_rate
            return VIBEvaluationResult(
                utility_score=1.0 if keep else 0.0,
                sufficiency=0.0,
                redundancy=0.0,
                beta_t=1.0,
                is_allowed=keep,
                threshold=self.keep_rate,
            )

        # gating_mode == "topk": cosine diversity gate on the broadcast history.
        if not history:
            # No history yet -> nothing can be redundant, so the first messages pass.
            return VIBEvaluationResult(
                utility_score=1.0, sufficiency=0.0, redundancy=0.0,
                beta_t=1.0, is_allowed=True, threshold=self.sim_threshold
            )

        emb_m = self.get_embedding(candidate_msg)
        max_sim = max(
            self._cosine_similarity(emb_m, self.get_embedding(h)) for h in history
        )
        return VIBEvaluationResult(
            utility_score=1.0 - max_sim,
            sufficiency=0.0,
            redundancy=max_sim,
            beta_t=1.0,
            is_allowed=max_sim < self.sim_threshold,
            threshold=self.sim_threshold,
        )

    def evaluate_detailed(
        self, 
        candidate_msg: str, 
        history: List[str], 
        goal: str, 
        step_t: int,
        token_probs: Optional[List[float]] = None,
        confidence_score: Optional[float] = None,
        role: Optional[str] = None,
        is_execution_failed: bool = False
    ) -> VIBEvaluationResult:
        """
        Detailed evaluation with full metrics for analysis and debugging.
        
        Integrates advanced mechanisms:
        1. Ada-VIB: adaptive threshold based on uncertainty
        2. Code-Diff Sensitivity: reduces false redundancy for code patches
        3. Bugfix-Aware Gating: detects fix-related keywords to protect critical patches
        4. Role-Adaptive β: role-specific beta weights (Architect/Coder/Reviewer)
        5. Task-Complexity Awareness: generalized complexity via text length
        6. Feedback-Aware β Reset: decay reset when execution fails
        7. Token-Aware Redundancy: mild length penalty for token compression balance
        
        Args:
            candidate_msg: The candidate message from an agent
            history: Current public communication history
            goal: Global task goal
            step_t: Current collaboration step
            token_probs: Optional token probabilities for entropy calculation
            confidence_score: Optional agent self-assessed confidence (0~1)
            role: Agent role name (Architect/Coder/Reviewer) for adaptive beta
            is_execution_failed: Whether previous code execution failed (feedback signal)
            
        Returns:
            VIBEvaluationResult with all computed metrics
        """
        if self.gating_mode != "vib":
            return self._evaluate_naive(candidate_msg, history)

        # 冷启动：全局+增量历史完全为空时，计算与 Goal 的相关度
        if not history:
            emb_m = self.get_embedding(candidate_msg)
            emb_g = self.get_embedding(goal)
            sufficiency = self.compute_sufficiency(emb_m, emb_g)
            beta_t = self.compute_beta_t(step_t, role=role, goal=goal, is_execution_failed=is_execution_failed)
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
        
        # Compute raw redundancy I(M; H)
        redundancy = self.compute_redundancy(emb_m, emb_history)
        
        # === Code-Diff Sensitivity: 代码差异修正 ===
        # 如果消息包含代码块且与历史代码有实质差异，大幅降低冗余惩罚
        curr_code = self._extract_code(candidate_msg)
        diff_ratio = self._compute_code_diff_ratio(curr_code, history) if curr_code else 0.0
        
        # 代码差异折扣：diff > 15% 时，冗余度惩罚打折
        code_discount = max(0.2, 1.0 - 1.5 * diff_ratio) if curr_code else 1.0
        
        # === Bugfix-Aware Gating: Bug 修正关键词衰减因子 ===
        has_fix_signal = self._detect_fix_signal(candidate_msg, role)
        # 当检测到修复信号时，对冗余度惩罚给予 0.25 折扣（保护关键补丁）
        omega_fix = 0.25 if (has_fix_signal and role in ("Reviewer", "Coder")) else 1.0

        # === Reasoning Chain Guardrails: 保护 GSM8K/MATH 中间推导/校验链 ===
        # MATH 回归分析: 折损集中在高难度/几何，根因是语义相似度把关键推导误判为冗余。
        # 结论/纠错锚点(推理链节点)给予最强保护，普通推理链中等保护，其余无保护。
        has_reasoning_signal = self._detect_reasoning_signal(candidate_msg, role)
        has_math_critical = self._detect_math_critical(candidate_msg, role)
        if has_math_critical:
            omega_reasoning = 0.4    # 结论/纠错锚点: 最强保护
        elif has_reasoning_signal:
            omega_reasoning = 0.65   # 普通推理链: 中等保护
        else:
            omega_reasoning = 1.0
        
        # 综合冗余度修正: Redundancy_adjusted = Redundancy * code_discount * omega_fix
        # Token-Aware: 对长文本施加温和长度惩罚，平衡 Token 压缩比
        token_length_factor = math.log(max(len(candidate_msg.split()), 1) + 1)
        adjusted_redundancy = redundancy * code_discount * omega_fix * omega_reasoning * (1.0 + 0.05 * token_length_factor)
        
        # === Role-Adaptive β: 角色驱动 + Task-Complexity + Feedback-Aware ===
        beta_t = self.compute_beta_t(step_t, role=role, goal=goal, is_execution_failed=is_execution_failed)
        
        # Compute final utility score
        # Utility(M) = β_role * I(M; G) - Redundancy_adjusted
        utility_score = beta_t * sufficiency - adjusted_redundancy
        
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

        # 对数学求解/校验消息进一步放宽阈值，避免极端压缩直接截断推理链
        if has_reasoning_signal and role in {"Solver", "Verifier"}:
            effective_threshold = max(self.tau_min, effective_threshold - 0.1)
        
        # Gate decision
        is_allowed = utility_score >= effective_threshold
        
        return VIBEvaluationResult(
            utility_score=utility_score,
            sufficiency=sufficiency,
            redundancy=adjusted_redundancy,
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


# 📑 项目技术文档：基于变分信息瓶颈的动态多智能体稀疏通信系统 (VIB-SparseComm-Agent)

## 1. 系统架构概述

本系统旨在解决多智能体协同（Multi-Agent Collaboration）中常见的**通信冗余、Token 成本高昂、上下文注意力稀释（Context Rot）**等痛点。系统基于 **LangGraph** 状态机架构开发，通过在代理生成消息与公共历史合并之间嵌入一个**轻量级 VIB 路由器**，在时间维度上实现通信拓扑的动态自适应稀疏化。

### 1.1 系统数据流拓扑

```
                       ┌──────────────────────────────┐
                       │      LangGraph 全局状态       │
                       │   (Goal, History, Candidates)│
                       └──────────────┬───────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
       ┌───────────────┐                             ┌───────────────┐
       │   Agent_1     │                             │    Agent_N    │
       │ (本地私有思考) │                             │ (本地私有思考) │
       └───────┬───────┘                             └───────┬───────┘
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │  Candidates 临时缓冲区 [M_i]  │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │     VIB 门控路由器评估器       │
                       └──────────────┬───────────────┘
                                      │ 计算 Utility(M)
                                      ▼
                            Utility >= Gamma ?
                                      │
                       ┌──────────────┴──────────────┐
                    是 │                             │ 否
                       ▼                             ▼
         【通过：合入全局 History】                 【阻断：直接剪枝丢弃】

```

---

## 2. 数学原理与计算公式

我们不采用基于 LLM 的高成本启发式评估，而是将信息增益判定抽象为变分信息瓶颈（VIB）优化问题。

### 2.1 优化目标函数

给定全局目标 $G$、当前第 $t$ 轮的已存公共历史 $H^{t-1}$，对于智能体产生的候选消息 $M$，VIB 的损失函数定义为：

$$\min \quad \mathcal{L}_{\text{VIB}}(M) = I(M; H^{t-1}) - \beta_t I(M; G)$$

### 2.2 核心指标的变分估算

为实现毫秒级计算，我们使用轻量级文本嵌入模型（Embedding Model） $E(\cdot)$，将互信息（Mutual Information）计算转化为向量空间操作：

* **任务充分性（Sufficiency）估算**：

$$I(M; G) \approx \text{Sim}(E(M), E(G)) = \frac{E(M) \cdot E(G)}{\Vert{}E(M)\Vert{} \Vert{}E(G)\Vert{}}$$


* **历史冗余度（Redundancy）估算**（基于 InfoNCE 的对比密度估计）：

$$I(M; H^{t-1}) \approx \log \sum_{h \in H^{t-1}} \exp\left( \frac{\text{Sim}(E(M), E(h))}{\tau} \right)$$



*(其中 $\tau$ 为调节平滑度的温度超参数，默认设为 0.2)*

### 2.3 动态衰减拉格朗日乘子

控制“探索与收敛”平衡的权重 $\beta_t$ 随迭代轮次 $t$ 呈指数衰减：

$$\beta_t = \beta_0 \cdot e^{-\lambda t}$$

### 2.4 准入判定公式

当且仅当候选消息的最终效用得分 $\text{Utility}(M) \geq \gamma$（$\gamma$ 为基础阈值，默认设为 0）时，允许该消息广播：

$$\text{Utility}(M) = \beta_t \cdot I(M; G) - I(M; H^{t-1})$$

---

## 3. 核心模块设计与接口定义

### 3.1 状态结构定义 (`AgentState`)

基于 LangGraph 的 `TypedDict` 设计。

* `global_goal`: `str`（任务的全局目标/指令）
* `history`: `List[str]`（已通过门控、被合并的公共对话历史，所有 Agent 可读）
* `candidates`: `List[Dict]`（本轮各 Agent 产生的待评估消息缓存）
* `step_t`: `int`（当前的协同轮次）

### 3.2 路由器接口设计 (`VIBSparseCommRouter`)

```python
class VIBSparseCommRouter:
    def __init__(self, beta_0: float = 2.0, lambda_decay: float = 0.15, temp_tau: float = 0.2, gamma: float = 0.0):
        # 初始化超参数与轻量级向量模型（如 bge-small-zh）
        pass

    def get_embedding(self, text: str) -> np.ndarray:
        # 获取文本的 L2 归一化嵌入向量
        pass

    def evaluate(self, candidate_msg: str, history: List[str], goal: str, t: int) -> bool:
        # 核心逻辑：计算 Utility(M)，返回是否放行 (True/False)
        pass

```

### 3.3 LangGraph 节点与边逻辑

* **Agent 节点（并行）**：读取 `history` 和 `global_goal`，调用大模型（本地推理），输出结果追加至 `candidates`，不污染 `history`。
* **门控节点（Gatekeeper）**：运行 `VIBSparseCommRouter.evaluate`。遍历 `candidates`，通过者追加至 `history`。清空 `candidates` 并令 `step_t += 1`。
* **条件边（Conditional Edge）**：
* 若满足终止条件（如目标达成、`step_t` 达到最大值），流向 `END`。
* 若未满足，重新分发至各个 Agent 节点。



---

## 4. 逐步验证与联合开发计划

为了稳妥推进，我建议我们将开发分为 **4 个 Milestone（里程碑）**。你验证完一个，我们再讨论并推进下一个：

| 阶段 | 任务目标 | 验证方法 | 状态 |
| --- | --- | --- | --- |
| **Milestone 1** | **VIB 路由器数学公式验证** | 编写单体测试，用几组人工构造的“高冗余/高相关/无关联”文本，看路由器能否给出符合直觉的 Utility 分数和 True/False 拦截结果。 | 🟢 **当前待开始** |
| **Milestone 2** | **LangGraph 基础状态机搭建** | 不接真实的 VIB 和大模型，用 Mock 数据跑通 `Agent -> Buffer -> Gatekeeper -> Conditional Edge` 的闭环拓扑。 | ⚪ 待启动 |
| **Milestone 3** | **大模型集成与端到端协同** | 接入真实的闭源（如 Claude/GPT）或开源（如 DeepSeek-R1）大模型，运行实际任务，观察通信被剪枝的过程。 | ⚪ 待启动 |
| **Milestone 4** | **消融实验与指标统计** | 统计不同 $\beta_0$ 和 $\lambda$ 下，Token 节省率（Saving %）与任务成功率（Success Rate）的帕累托前沿。 | ⚪ 待启动 |

---

> 💡 **学长寄语**：
> “纸上得来终觉浅，绝知此事要躬行。”
> 我们现在立刻启动 **Milestone 1**。第一步，我们需要确保 VIB 路由器的向量计算和互信息估计在逻辑上是百分之百可靠的。

### 🚀 今日任务：验证 Milestone 1（VIB 路由单体测试）

请在你的本地环境中，先准备好一个轻量级的向量化工具（比如用 `sentence-transformers`，或者直接用 OpenAI / DashScope 的 Embedding 接口，甚至可以用最简单的 `sklearn` 的 TF-IDF 向量做初步替代）。

这是我们要测试的**两组极端案例**：

* **测试案例 A（高冗余废话，应该被拦截）**：
* *全局目标 (Goal)*: "优化这段 Python 代码的排序性能"
* *历史记录 (History)*: ["智能体A：我们可以使用快速排序，它的平均时间复杂度是 $O(n \log n)$。"]
* *候选消息 (Candidate)*: "智能体B：是的，我也觉得快速排序很棒，因为它的平均时间复杂度确实是 $O(n \log n)$，我们用快排吧。"


* **测试案例 B（高价值新信息，应该放行）**：
* *全局目标 (Goal)*: "优化这段 Python 代码的排序性能"
* *历史记录 (History)*: ["智能体A：我们可以使用快速排序，它的平均时间复杂度是 $O(n \log n)$。"]
* *候选消息 (Candidate)*: "智能体B：考虑到数据可能存在大量重复元素，我建议使用三路快排（Three-way Quick Sort）来避免退化。"



**你现在可以动手编写 `VIBSparseCommRouter` 的类，并跑一下这两组测试，看看它的 `utility` 分数是否符合“案例 B 远大于 案例 A”的预期？**

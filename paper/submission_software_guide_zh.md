# 面向多智能体大语言模型的轻量消息门控：通信—精度权衡的实证研究

**作者：** 屈占祥（独立研究者，ORCID: 0009-0000-0137-9751）

---

## 摘要

多智能体大语言模型系统通常让编码、测试、验证等角色通过共享历史广播每一条候选消息，但并非所有消息都对最终答案有贡献，盲目广播既带来输入上下文的通信开销，也可能引入错误建议。针对"团队是否需要广播全部候选消息"这一问题，本文以信息瓶颈思想为动机，设计了一种确定性的轻量消息门控策略 VIB-Sparse，在候选消息生成之后、历史合并之前，依据候选与题目目标的相关性以及候选与已有历史的冗余度进行准入判断。本文在 HumanEval（164 题）、GSM8K（1319 题）和 MATH（5000 题）三个基准上系统分析了通信量与任务表现之间的权衡。结果表明：与完整通信相比，VIB-Sparse 在三个基准上的广播消息量分别减少 21.14%、30.49% 和 25.59%，而精度差分别为 −1.83、−0.08 和 −0.92 个百分点；但 GSM8K 与 MATH 记录的总 Token 减少仅为 4.96% 与 4.42%，远小于消息数下降幅度。在匹配消息预算的对照中，简单余弦阈值门控与 VIB-Sparse 的精度点估计均为 97.00%，配对差值区间为 [−2.50, +2.50] 个百分点，随机门控亦仅低约 1 个百分点。上述结果共同表明：生成后门控可以稳定降低通信量，但其收益主要体现为"节省通信"而非"提升精度"，现有证据不足以支持 VIB 机制相对于简单门控的独有精度优势。本文进一步审计了评分口径、参数透传与费用关联等实现细节，厘清了可报告结论与历史探索证据的边界。

**关键词：** 多智能体大语言模型；稀疏通信；信息瓶颈；门控；成本评估；实验可复现性

---

## 1 引言

随着大语言模型能力的提升，将软件开发与数学推理分解为多个智能体协同完成已成为一类重要的系统范式。ChatDev、MetaGPT、AgentCoder、SWE-agent 等代表性工作分别以不同方式划分编码、测试、评审等角色，并通过共享历史在智能体之间传递候选消息[1-4]。这类架构的一个隐含假设是：更多的讨论与更完整的上下文通常有利于最终答案。然而，消息进入上下文并不代表它对最终正确性有实际贡献，冗余或错误的建议同样会被广播，既增加后续合成阶段的输入开销，也可能干扰决策。

本文区分两个容易混淆的问题。第一是**上下文质量**问题：删除重复或错误内容能否改善答案，这需要在控制候选内容的前提下进行因果实验，而不能仅凭相似度分数推断。第二是**广播与实际费用**问题：候选消息一旦生成，其生成费用便已发生，生成之后将其丢弃只能减少后续可能使用的输入上下文，无法撤销已经发生的请求费用。将这两个问题混为一谈，往往会高估门控的收益。

基于以上区分，本文围绕三个具体问题展开研究。其一，在代码与数学两类任务上，生成后门控能够以多大的精度代价换取多少广播消息的节省？其二，与随机门控、余弦阈值门控等简单基线相比，VIB-Sparse 的复杂门控是否带来额外收益？其三，消息数下降、输入 Token 下降与完整费用下降三者之间是否一致，报告通信收益时应当以何种口径为准？

针对上述问题，本文以信息瓶颈（Information Bottleneck, IB）思想为动机，提出并评估了一种轻量消息门控策略 VIB-Sparse。需要强调的是，该方法属于确定性的嵌入启发式门控，其名称不表示已经拟合、优化或证明了变分互信息下界。本文的贡献可归纳为三点：

（1）在三个不含参考辅助早停的主基准上报告了精度与广播量之间的权衡，并将依赖参考测试参与早停的历史 MBPP 结果单独隔离为 oracle-only 证据。

（2）在相同任务、相近消息预算下，将随机门控、余弦阈值门控与 VIB-Sparse 进行对照，明确不把"消息数匹配"误称为"总费用匹配"。

（3）对评分、参数透传、合成刷新与成本关联等实现环节进行审计，在保留真实历史正收益的同时，区分探索性观察与稳定方法优势。

本文的立场是审慎的实证研究：既如实报告 VIB-Sparse 在通信量上的可观节省，也如实披露其在精度上未表现出相对于简单门控的稳定优势，并据此给出关于协议设计与评估口径的建议。

---

## 2 相关工作

多智能体软件工作流方面，ChatDev 通过角色化的对话流程完成软件开发[1]，MetaGPT 引入标准操作流程与结构化输出[2]，AgentCoder 使用迭代测试与优化提升代码生成质量[3]，SWE-agent 提出智能体—计算机接口以自动化软件工程任务[4]，OpenDevin 则提供了面向通用智能体的开放平台[5]。这些工作共同确立了角色分工与共享历史在多智能体系统中的重要地位，但大多默认广播所有候选消息，较少显式讨论通信量的取舍。

通信裁剪与提示压缩方面，DyLAN 通过动态团队优化调整智能体协作结构[6]，LLMLingua 通过提示压缩加速大语言模型推理[7]，AgentPrune 提出经济化的多智能体通信管线[8]，AgentDropout 通过动态消除智能体以在保持性能的同时降低 Token 开销[9]。这些工作表明，稀疏通信与提示压缩本身并非本文的独创，本文的定位是在统一评测框架下，对"门控收益究竟来自通信节省还是精度提升"这一具体问题给出审慎的实证回答，而非声称稀疏通信的新颖性。

理论动机方面，信息瓶颈原理与深度变分信息瓶颈为信息保留与正则化提供了理论或训练框架[10,11]，InfoBERT 则从信息论视角提升语言模型的鲁棒性[12]。本文仅借用信息瓶颈的权衡思想作为启发式动机，其实现并不估计真实分布下的互信息，也不优化相应的变分界，理论强度与上述工作不同。此外，Self-Refine 提出了迭代自反馈的强单智能体对照范式[13]，关于多智能体辩论评估的研究则强调推理预算与模型差异对结论的影响[14]，这些都为本文设置对照实验提供了方法参考。StreamingLLM 与 MemWalker 主要关注长上下文与记忆导航[15,16]，与本文所研究的短团队广播门控并不等同。

---

## 3 方法

### 3.1 任务与工作流

本文在代码生成与数学推理两类任务上开展实验。代码任务使用 HumanEval[17]，以可执行代码测试作为评判依据；历史 MBPP 结果[18]因涉及参考测试参与早停而被隔离，不作为主结果。数学任务使用 GSM8K[19] 与 MATH[20]，其中 MATH 的答案形式包含分数、表达式、列表、元组与区间等，不能简单取最后一个数字进行评判。

为刻画不同通信策略的影响，本文设置三种工作流。**Vanilla** 为单智能体合成，直接接收题目而不接收团队历史，作为较低推理预算的锚点，但不是等费用的单智能体对照。**Full** 在代码任务中使用 Coder、Tester、Reviewer 三个角色，在数学任务中使用 Planner、Solver、Verifier 三个角色，每轮各角色产生一个候选，同轮采用冻结历史，所有候选合并广播，讨论后由合成器生成最终答案。**VIB** 使用与 Full 完全相同的角色、提示词函数与评测器，区别仅在于候选生成后、历史合并前加入门控：只有通过准入判断的消息才进入公共历史。需要说明的是，Full 与 VIB 是分开的随机执行，二者并不共享完全相同的候选流，因此本文的对照是策略级对照，而非对同一候选流的因果干预。

图 1 给出了工作流的概念边界。评分材料（参考答案与隐藏测试）仅用于输出完成后的评测，不进入部署决策；门控日志记录的是分数与准入结果，而非合成器的注意力，因此不能证明某条消息对最终答案存在因果贡献。

### 3.2 信息瓶颈动机与实现代理

令 G 表示题目目标，H 表示历史，M 表示候选消息，理想的信息瓶颈权衡可写为

L_IB(M) = I(M; H) − β_t · I(M; G)

其中 I(·;·) 表示互信息，β_t 为权衡系数。信息瓶颈与深度变分信息瓶颈提供的是理论或训练框架[10,11]；本文实现并不估计真实分布下的互信息，也不优化相应变分界，而是采用冻结的 all-MiniLM-L6-v2 编码器（384 维）构造启发式代理。目标相关性代理为候选与题目的嵌入余弦相似度

S(M, G) = cos(E(M), E(G))

历史冗余代理采用长度归一化的 log-sum-exp 形式

R(M; H) = (1/√|H|) · log Σ_{h∈H} exp( cos(E(M), E(h)) / τ )

其中 E(·) 为嵌入编码，τ 为温度参数。归一化用于避免历史长度直接导致分数饱和，但这并不证明该分数等于真实的冗余信息量。代码任务还使用候选与最后历史代码块之间的 SequenceMatcher 差异比例 δ 进行折扣，折扣因子为 max(0.2, 1 − 1.5δ)。

最终的门控效用函数为

U(M) = β_t · S(M, G) − R_adj(M)

其中 R_adj(M) 在冗余代理基础上乘以代码差异、纠错关键词、推理关键词等修正因子，并附加消息长度的对数项。门控权重与自适应温度分别定义为

β_t = (β_role + complexity_bonus(G)) · e^(−λ·t_eff)
τ_adaptive = max(τ_min, γ − α·H̄(Y))

需要强调的是，上述各项均为启发式修正，而非可靠性估计器：较大的文本改动并不一定是正确修复，纠错关键词的出现也不等于修复有效。本文主配置为 β_0=3.2、λ=0.08、τ=0.5、γ=−0.4、α=0.35、τ_min=−1，关闭数学角色权重覆盖与推理保护。由于构造器默认值与主配置不同，复现时必须显式传参；原始 gate 决策与经过角色救援后的广播决定也必须分开统计。

还需进一步明确门控轨迹的因果边界：门控日志记录的是分数与准入结果，既不记录合成器的注意力，也不能证明某条消息对最终答案存在因果贡献；合成器接收的是消息文本而非门控分数。若某题的所有候选均被广播，就不能把该题的成功归因于"过滤掉了错误建议"。这一边界决定了本文所有解释的限度，也是后续开展因果验证实验时必须直接面对的问题。

### 3.3 模型、评分与统计

主实验使用滚动别名 deepseek-chat，旧产物无法恢复不可变的服务快照。调用层对 Agent 显式设置温度 0.2、上限 256，对代码合成设置 0.1/1024，对数学合成设置 0.1/256；由于这些参数覆盖客户端默认值，文件名中的 t0 并不证明真实温度为 0，因而本文不声称温度稳健性。

评分方面，HumanEval 通过执行 check(candidate) 判断正确性；MBPP 使用顶层断言且不传 HumanEval 的入口点。旧版合成刷新曾只定义测试函数而未调用，产生 100/100 的假象，独立重评后主实验 HumanEval 为 151/164 与 148/164。MATH 采用独立的 whole_answer_v1 匹配器，优先提取最后一个平衡的 boxed 答案，再提取显式答案标签或整条输出，不回退到最后数字；对有序列表、元组、区间开闭进行显式保留，对集合与并集进行无序比较，常见分数、根式与代数表达式由受限 AST 与 SymPy 处理，未知格式保守地整文本比较或判为失败。该匹配器并非完整的数学语义 oracle。

统计方面，单比例区间使用 Wilson 方法，差值使用 10000 次任务配对 bootstrap。本文没有预注册非劣效界值，因此区间跨零不意味着无损或等效；重复合成先按任务聚合，不把模型调用数当作独立任务数。

---

## 4 实验结果与分析

### 4.1 主结果

表 1 给出三种方法的历史表现与通信预算。就精度而言，HumanEval 上 VIB 相对 Full 少 3 题，差值 −1.83 个百分点，95% 区间为 [−6.71, +3.05]，McNemar 检验 p=0.607；GSM8K 上差值 −0.08 个百分点，区间 [−0.83, +0.68]。这些是未解决的差异，而非已通过非劣效检验。MATH 上匹配率为 78.58% 与 77.66%，差值 −0.92 个百分点，区间 [−1.78, −0.04]，p=0.045，该负差虽在名义显著性水平上不为零，但未经多重检验校正，且对评分覆盖口径敏感，不能据此作普遍结论。

就通信量而言，VIB 相对 Full 在三个基准上的广播消息量分别减少 21.14%、30.49% 与 25.59%，这是一致且可观的节省。然而，GSM8K 与 MATH 记录的总 Token 减少仅为 4.96% 与 4.42%，HumanEval 的刷新代码沿用旧费用字段，其 3.88% 只能视为历史计数差，不能视为刷新后的完整费用节省。这一显著落差揭示了一个关键事实：消息数下降与总费用下降之间存在不可忽视的鸿沟，候选生成费用占据主导，门控只能削减后续输入侧的开销。

| 数据集 | 题数 | Vanilla (%) | Full (%) | VIB (%) | Δ(VIB−Full) (pp) | 95% CI | 广播减少 | Token 减少 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HumanEval | 164 | 93.90 | 92.07 | 90.24 | −1.83 | [−6.71, +3.05] | 21.14% | 3.88% |
| GSM8K | 1319 | 59.29 | 95.30 | 95.22 | −0.08 | [−0.83, +0.68] | 30.49% | 4.96% |
| MATH | 5000 | 46.16 | 78.58 | 77.66 | −0.92 | [−1.78, −0.04] | 25.59% | 4.42% |

**表 1　三种方法的历史表现与通信预算**（MATH 为保守完整答案匹配率；区间跨零不证明等效或非劣效）

相对历史 Vanilla 而言，Full 在 GSM8K 与 MATH 上分别高 36.01 与 32.42 个百分点，VIB 分别高 35.93 与 31.50 个百分点，而 HumanEval 上没有同样的优势。由于团队使用了更多推理预算，这些差值不能用于隔离多智能体结构本身的收益。

### 4.2 简单门控与跨模型对照

为检验复杂门控是否优于简单规则，本文在一个 200 题的 GSM8K 子集上进行了消融与对照。关闭角色权重时，广播减少 30.25%，差值 −0.50 个百分点，区间 [−1.50, 0.00]；开启角色权重时，广播减少 25.17%，差值 0.00。由于不同随机运行及 Full 精度的自然波动，该结果不能证明选择性的角色收益。

在匹配 VIB 消息预算的条件下，余弦阈值 0.75 门控减少 29.50% 的广播，VIB 减少 30.17%，两者精度均为 97.00%，配对差值区间 [−2.50, +2.50] 个百分点；随机门控减少 31.50% 的广播，精度 96.00%，相对 VIB 差 −1.00，区间 [−4.00, +1.50]。同时，随机门控、VIB 与余弦门控的记录 Token 减少分别为 13.08%、5.08% 与 4.72%，这再次表明消息预算接近并不等于费用相等，不能据此作等费用的优越性判断。

在跨模型迁移观察中，DeepSeek 参考行与 Qwen 行相对 Full 的精度差分别为 +0.5 与 +1.5 个百分点，消息减少均超过 25%；其中 Qwen 为 97.50% 与 99.00%，区间 [0.00, +3.50]，p=0.25。这些属于有限迁移观察，不构成模型无关保证。

### 4.3 历史正收益与实现缺陷

本文对早期三个 HumanEval 正收益批次重新执行了原入口点测试，标签完全保持不变：CLI 修正版 Full/VIB 为 152/155 题（92.68%/94.51%），差值 +1.83，区间 [−1.83, +5.49]，p=0.508；仲裁规则版为 148/154 题（90.24%/93.90%），差值 +3.66，区间 [0.00, +7.32]，p=0.109；最少角色救援版为 144/149 题（87.80%/90.85%），差值 +3.05，区间 [−1.83, +7.32]，p=0.302。加上刷新前主实验，共 1312 次存档代码检查未出现标签翻转。这些结果说明历史正收益是真实的，但各批次使用相同的 164 题，不能合并制造独立样本，也不能选择最好的一次来替代当前主表；最高 VIB 相对历史 Vanilla 也仅多 1 题。

审计还确认了两类实现缺陷。其一，合成刷新仅修改 output 与 passed 字段，通过复制旧 variant 保留 Token 字段，导致 94 份变化输出继承旧账目，因此不能声称刷新后的完整费用节省。其二，零 API 测试截获到客户端显式参数覆盖了默认温度，因而必须撤回"真实温度为零"的稳健性表述。这些缺陷不改变已有输出得分，也不自动解释历史正收益，但提示在报告通信与费用收益时须格外谨慎。

### 4.4 接收端重放与保护配置

为检验历史信息对最终合成的作用方向，本文进一步开展了接收端重放实验：在固定候选流的前提下，向合成器分别提供完整历史、各门控后的历史、仅首轮历史以及无历史等多种策略，观察最终答案的变化。在两个各 24 题的历史子集上，HumanEval 六种策略的精度依次为 83.33%、87.50%、87.50%、85.42%、91.67%、91.67%，其中完整历史相对仅首轮的差值为 −8.33，区间 [−20.83, 0.00]；MATH 中所有接收历史的策略均为 87.50%，而无历史仅为 39.58%。这些结果表明，完整历史在个别情形下可能引入错误，但已见任务与较小样本量不支持"协作普遍带来负收益"的结论，首轮历史在 MATH 上的贡献则十分显著。

此外，配置取舍实验表明，在相同 5000 题、相同完整答案匹配下，同时开启数学角色权重与推理保护的旧批次为 78.46%/78.32%，差值 −0.14，区间 [−1.04, +0.72]，消息减少 5.06%、Token 减少 0.48%；主配置关闭保护时为 78.46%/77.42%，消息减少 25.39%、Token 减少 4.21%。可见保护配置以更小的压缩换取接近 Full 的精度，而主配置则以更大的通信节省换取微小的精度代价，二者构成明确的取舍；但由于两批并非固定共同候选实验，不能把差值全部归因为保护机制的因果收益。

### 4.5 讨论与局限

综合上述结果，本文能够支持的核心结论是：**生成后门控可以在精度基本持平的前提下稳定减少广播消息量，但该收益主要体现为"省通信"而非"提精度"**。VIB 保留了较多的团队效果，历史正收益是真实观察，但复杂门控尚未稳定超越简单规则；这既不证明 VIB 工作没有价值，也不证明恢复某个历史版本就能获得可靠的方法优势。

本文的主要局限包括：滚动模型别名导致旧产物不可复现；团队规模较小（三角色）；任务经反复调参而存在已见偏差；Vanilla 并非等费用对照；存在事后选择；MATH 语义匹配覆盖不足；温度参数透传缺陷；刷新费用缺失；历史 MBPP 存在 oracle 早停；以及多重检验问题。这些局限意味着，区间跨零不能视作无损，显著负差也不能脱离评分覆盖与配置差异而作普遍结论。

---

## 5 结论

本文围绕"多智能体大语言模型团队需要多少通信"这一问题，对信息瓶颈启发的轻量消息门控策略进行了系统的实证评估。结果表明，门控能够在三个主基准上以较小的精度代价换取 21%–31% 的广播消息节省，但总 Token 节省远小于消息数下降，且简单门控在匹配预算下与 VIB 表现接近。据此，本文的结论是审慎的：生成后门控的价值在于节省通信，而非带来独有精度优势。这一结论对多智能体系统的协议设计与评估具有直接参考意义——在报告通信收益时，应当区分消息数、输入 Token 与完整费用三个层次，并避免以消息数匹配替代费用匹配。后续工作应优先修复费用关联与可复现性，开展固定候选流的单因素对照实验，并引入预算受限的等费用验证，以在更可靠的基础上刻画通信与精度的真实权衡。

---

## 参考文献

[1] Qian C, Liu W, Liu H, et al. ChatDev: Communicative Agents for Software Development[C]//Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), Long Papers. 2024.

[2] Hong S, Zhuge M, Chen J, et al. MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework[J]. arXiv preprint arXiv:2308.00352, 2023.

[3] Huang D, Zhang J M, Luck M, et al. AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation[J]. arXiv preprint arXiv:2312.13010, 2023.

[4] Yang J, Jimenez C E, Wettig A, et al. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering[C]//Advances in Neural Information Processing Systems (NeurIPS). 2024.

[5] Wang X, Li B, Song Y, et al. OpenDevin: An Open Platform for AI Software Developers as Generalist Agents[J]. arXiv preprint arXiv:2407.16741, 2024.

[6] Liu Z, Zhang Y, Li P, et al. Dynamic LLM-Agent Network: An LLM-agent Collaboration Framework with Agent Team Optimization[J]. arXiv preprint arXiv:2310.02170, 2023.

[7] Jiang H, Wu Q, Lin C Y, et al. LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models[C]//Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2023: 5097-5110.

[8] Zhang G, Yue Y, Li Z, et al. Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems[J]. arXiv preprint arXiv:2410.02506, 2024.

[9] Wang Z, Wang Y, Liu X, et al. AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration[C]//Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics. 2025: 24013-24035.

[10] Tishby N, Zaslavsky N. Deep Learning and the Information Bottleneck Principle[C]//IEEE Information Theory Workshop (ITW). 2015.

[11] Alemi A A, Fischer I, Dillon J V, et al. Deep Variational Information Bottleneck[C]//International Conference on Learning Representations (ICLR). 2017.

[12] Wang B, Wang S, Cheng Y, et al. InfoBERT: Improving Robustness of Language Models from an Information Theoretic Perspective[C]//International Conference on Learning Representations (ICLR). 2021.

[13] Madaan A, Tandon N, Gupta P, et al. Self-Refine: Iterative Refinement with Self-Feedback[C]//Advances in Neural Information Processing Systems (NeurIPS). 2023.

[14] Zhang H, Cui Z, Chen J, et al. Stop Overvaluing Multi-Agent Debate -- We Must Rethink Evaluation and Embrace Model Heterogeneity[J]. arXiv preprint arXiv:2502.08788, 2025.

[15] Xiao G, Tian Y, Chen B, et al. Efficient Streaming Language Models with Attention Sinks[J]. arXiv preprint arXiv:2309.17453, 2023.

[16] Chen H, Pasunuru R, Weston J, et al. Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading[J]. arXiv preprint arXiv:2310.05029, 2023.

[17] Chen M, Tworek J, Jun H, et al. Evaluating Large Language Models Trained on Code[J]. arXiv preprint arXiv:2107.03374, 2021.

[18] Austin J, Odena A, Nye M, et al. Program Synthesis with Large Language Models[J]. arXiv preprint arXiv:2108.07732, 2021.

[19] Cobbe K, Kosaraju V, Bavarian M, et al. Training Verifiers to Solve Math Word Problems[J]. arXiv preprint arXiv:2110.14168, 2021.

[20] Hendrycks D, Burns C, Kadavath S, et al. Measuring Mathematical Problem Solving with the MATH Dataset[J]. arXiv preprint arXiv:2103.03946, 2021.

---

*基金项目：无。作者简介：屈占祥，独立研究者，研究方向为大语言模型多智能体系统与稀疏通信。投稿系统中作者简介栏目的出生年、性别、学历等字段，请按期刊要求另行填写。*

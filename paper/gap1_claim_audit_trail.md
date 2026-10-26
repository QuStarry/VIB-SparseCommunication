# 阶段性论文审计与下一阶段研究路线

更新日期：2026-10-23。当前论文事实源是 `paper/paper.tex` 和生成表格对应的 JSON。
本次只改写、审计和运行本地验证，没有启动新付费实验，也没有改动核心路由。

## 当前定位

标题：**How Much Communication Do LLM Teams Need? An Empirical Study of Lightweight Message Gating**。

阶段性贡献是比较通信预算与推理成本、披露简单门控对照和负向结果，不再以“VIB 机制优越”或“所有基准零损失”为结论。
这是一份可继续完善的实证稿，不等于已具备任何会议的录用保证或正式投稿条件。

| 数据源 | 当前可支持的结论 | 不能推出的结论 |
|---|---|---|
| `results/main_with_math_merged.json`，HumanEval | 两臂均 164/164；消息少 21.14%，记录 tokens 少 3.88% | 总体误差为零、已验证预算再投资 |
| 同上，GSM8K | 1319 题，消息少 30.49%，tokens 少 4.96%，精度差 -0.08pp | 已通过预注册非劣效检验、VIB 特有收益 |
| 同上，MATH | 5000 题，消息少 25.59%，tokens 少 4.42%；精度差 -0.92pp，CI [-1.78,-0.04] | 精度零损失、负向结果属于无害噪声 |
| 同上，MBPP | 257 题，精度差 +0.78pp；消息和 tokens 分别增加 0.12%、0.05% | 全部基准都压缩成功 |
| `results/gating_sweep/`，200 题对照 | VIB 和余弦阈值的精度点估计均 97%；配对 CI [-2.50,+2.50]pp | 方法等价、VIB 价值被证伪、简单门控普遍足够 |
| 同上，随机门控 | 消息少 31.50%，tokens 少 13.08%，精度 96%；对 VIB 的 CI 跨零 | 匹配消息数就是匹配成本 |
| 主结果 MBPP/260、285 trace | 两题 Full 失败、VIB 成功，但每臂均广播 6/6 | 评分传给合成器、过滤错误消息导致成功 |

## 本轮查漏补缺

- 已移除不可追溯的案例效用数值与“合成器按冗余评分分配注意力”的表述。历史两例没有 `gate_decisions`；当前 serializer 存在不代表历史证据存在。
- 已把互信息“下界”改为 embedding surrogate。代码没有拟合可验证的变分界；论文效用公式现在包含实际的历史归一化和代码、纠错、推理、长度折扣。
- 已撤回未预设界值的“正式非劣效”结论。CI 跨零表示未检出差异，不是等价证明；四次 Full 的 2.5pp 极差不是置信区间。
- 已注明 Full/VIB 分开随机执行，Vanilla 不是等算力对手；“多智能体 +36.01pp”不能排除更多推理预算的贡献。
- 已注明 MBPP OR 早停分组依赖结果。当前 probe 是一轮团队协作加合成，并由参考评测器决定是否重跑，不能称为部署时可用的 one-shot 早停。
- 已披露重合成历史：主结果 JSON 标记 `refreshed_synthesis=True`，并非统一快照的新一批端到端实验。
- 仍需修复新实验账目：MBPP probe 失败后当前返回重跑分支的 token counters，可能漏计被丢弃 probe。历史账目不可凭空补齐；新运行须按请求累加。
- 仍需核对 rescue 日志：`src/workflow.py` 当前按被救援数量近似标记前若干 rejected 条目，可能标错具体消息；新因果实验前要按消息身份准确记录。
- 已修正 AgentCoder、GSM8K、MemWalker 的书目；删除指向不相关代数学论文的 RECENT 项；补 AgentDropout 和预算评估相关工作。完整逐条书目复核仍是投稿前任务。
- 正式 venue 未确定，当前为自定义 article 版式，不能将编译成功视为模板合规。

## 下一阶段：三个方向，分批验证

### 方向 A：关键纠错保留，优先做因果机制验证

问题：一条与历史很相似的消息，可能只改一个数字、运算符或边界条件，却决定答案是否正确。余弦冗余不等于任务价值。
这是待验证假设，不是已发现 VIB 的优势；现有 VIB 也可能失败。

零 API 自然轨迹审计已执行：MATH 有 265 个 Full-only 与 219 个 VIB-only，净少 46 题。
265 个回退任务都保留当前格式 gate 日志，其中 193 题广播数减少，72 题未减少；不能把全部回退都归因于删消息。
Geometry 有 479 题、Full-only 37 / VIB-only 23，描述性差值 -2.92pp；
Level 5 有 1324 题、116 / 96，差值 -1.51pp。
类别和难度切片重叠且是事后检查，未做多重比较修正，不可作为确认性结论。
这提供了比“强行扩大原有 +2/21 案例”更自然的探索起点：审计错误删除与正确纠错之间的关系，而非预设 VIB 是赢家。

第二轮零 API 审计（评分口径与删减归因，2026-10-23）：在上一段 193/72 的基础上去掉评分伪影后，把 265 个 Full-only 回退拆成三类——
11 题（4.2%）的 VIB 输出在鲁棒规范化后与参考答案等价，属评分口径伪影；
71 题（26.8%）两臂广播消息数相同（Δ=0），但消息内容可以不同，不能直接归为采样噪声或排除门控影响；
183 题（69.1%）VIB 广播数更少，才是需要因果实验验证的部分。
反方向也有 6 个 VIB-only 题属评分伪影。机制来源是 `MathEvaluator._extract_last_number`：
正则 `[-+]?\d[\d,]*` 对含逗号的多值答案（483/5000≈9.7%）会把 `12, 10, 6` 与 `12,10,6`
解析成 `6` 与 `12106`，同一答案因空格差异判定相反。
上述规范化只是候选筛查，不是经过验证的数学等价评测：任意删除幂号会合并不同表达式，逗号重排也可能破坏有序元组语义。
因此撤回此前“约 41% 回退是评分伪影”和“可排除约 31% 门控责任”的推断；11/71/183 只保留为历史探索分桶，不作为确认性归因或重评分。
零 API 不能证明「关键纠错误删」：
两臂是独立随机执行，VIB 的 history 不是 Full history 的子集，缺少共同候选流，
因此真正的因果验证必须靠冻结候选流的受控实验。

1. 先用已有历史做零 API 审计，查找真实的代码小改动和数学纠错，并统计被各门控误删的情况。只作发现集，不用于确认性显著性结论。
2. 准备与发现/调参数据不重叠的候选任务池，先冻结 task IDs 和校验哈希；建议 20 题校准、60 题探索评估。若池子无法验证不重叠，则明确降级为探索，不称 held-out。
3. 60 题包含自然轨迹；受控的重复/正确纠错/错误纠错扰动仅作辅助机制测试，不能代替自然任务表现。参考答案只给 evaluator，不给门控或合成器。
4. 对每个任务冻结共同候选文本；比较 Full、Random、Cosine threshold、VIB 四种接收历史。各方法 3 次独立合成，共 720 次评估合成调用（不含校准、候选生成和重试）。
5. 同一任务的重复合成按任务聚类分析；720 次调用不是 720 个独立问题。指标包括关键纠错保留率、错误纠错放行率、Pass@1、tokens 与延迟，所有方法获得相同证据。
6. 先看是否能在自然任务上观测到纠错误删，而不是用特殊合成例强行造 VIB 优势。60 题仅作探索；若趋势不清楚或 VIB 不优于强简单规则，停止该方法主张，保留负结果。

潜在方法增量：验证器证据或依赖关系支持的消息保留，而非重新包装已有余弦目标项。
新颖性尚未确认，须补全 verifier-aware / evidence-aware 路由相关文献后再立方法名。

### 方向 B：从“生成后丢弃”到“生成前调度”

问题：被丢弃的消息已经付费生成，能否事先跳过低价值调用，并保护关键验证步骤？

先冻结 3-agent 工作流，比较 Full、post-generation gate、固定轮次裁剪、random agent skip 与新调度器；再测试 6-agent / 3-round 的迁移。
所有调度器只能使用当时可见证据，不得用参考答案早停；统计调度器自身 API 成本。
总 input/output tokens、美元成本、P50/P95/P99、429、成功率须一起报，确认性阶段按 task × model 配对。
建议候选扩展门槛为总 tokens 节省至少 15%、探索精度差不低于 -1pp；这是尚待批准的筛选标准，不是现有结果或正式非劣效结论。
在独立评估前冻结可接受精度损失界值和成本目标；不能在看到结果后扩大界值。

### 方向 C：等预算单智能体与异构团队

比较预算受限的 single-agent self-consistency/self-refine、同构团队、异构团队及简单门控。
最终任务成功率优先，预算用实际请求 input/output tokens 或同一货币费用，不只算消息或 thinking tokens。
候选选择器和 verifier 的调用计入总账；禁止用隐藏参考评测结果挑最佳候选。
先做一个模型/任务块，再迁移第二模型，不把换模型与换路由同时作为唯一“改进”。
该方向是所有 multi-agent 主张的必备对照，不能保证产生正结果。

## 成本与统计：更正此前评估

此前把外推 token/耗时称为“全部实测”、给出未经验证的功效样本量是不准确的。
实测只覆盖已经完成的运行；新场景时间受串行依赖、并发、重试和输出长度影响。
MBPP 重复 21 题可以改善对这 21 题采样分布的估计，但不会增加独立任务覆盖；也不能断言任何重复方案必然“无法判定”。
MBPP 全量重跑后的 MR 分组可能变化，不能保证仍为原来 21 题。

用户于 2026-10-23 授权按接近 DeepSeek 官方价格推进合理的小规模实验，优先研究收益，而非继续等待精确账单单价。
官方价格页当日 Flash 峰时输入未命中 $0.30/百万 tokens、输出 $1.20/百万 tokens；
实验采用此不享受缓存折扣的代理计价，并设首轮 $3 上限。私有网关实际账单仍需另核，不把代理金额冒充实付费用。
计费公式为 `(billable_input_tokens * input_rate + billable_output_tokens * output_rate) / 1e6`；
缓存命中、工具调用或不同模型须按各自单价另计，失败重试也需账单对照。

| 已完成 200 题运行 | 两臂 input tokens | 两臂 output tokens | 总记录 tokens |
|---|---:|---:|---:|
| VIB anchor | 758636 | 145438 | 904074 |
| Random | 719925 | 141542 | 861467 |
| Cosine 0.75 | 756347 | 142442 | 898789 |

这些数值来自相应 `detailed_results.gsm8k` 的 prompt/completion 字段逐题求和，只是历史标定，不是新任务报价。
方向 A 的 720 次合成若统一选择数学输出上限 256，正常评估输出上限为 184320 tokens；不包括输入、校准、候选生成、重试。
执行时必须提供请求数、累计费用和重试硬上限；达到预算即停止，已完成请求可从缓存幂等续跑。
方向 B/C 未做新任务 smoke，不能沿用短轮 GSM8K 的时间和费用系数报价。

配对功效取决于不一致任务比例而非单独的准确率。当前 Random/VIB 分歧为 8/200=0.04，Cosine/VIB 为 6/200=0.03。
仅在假定未来分歧率仍为 0.04、真实差异为 1pp、独立任务、双侧 α=0.05、power=0.8 的正态近似下，
`N ≈ (1.96+0.84)^2 * 0.04 / 0.01^2 ≈ 3136`；这不是实际新实验功效证明，更不是必须投入的目标。
因此先用小实验识别效应与失败模式，再做按任务聚类的功效模拟；不为追显著性盲目扩样。

## 执行顺序与交付边界

- [x] 阶段一：重定位标题、摘要和贡献；修正方法公式、案例因果主张、统计和成本披露。
- [x] 表格生成器新增完整配对任务集合与重复 task ID 守卫；表格由脚本生成。
- [x] 本地验收：61 项测试通过；六张表二次导出逐字节一致；`hfuzz=0pt` 下 Overfull/undefined/error 均为 0；PDF 共 13 页。当前自定义版式不代表任何 venue 合规。
- [ ] 新实验前：请求级账本、精确 rescue 标记、独立评估池、不可见参考答案隔离、模型响应元数据与配置哈希。
- [x] 方向 A 首次零 API 审计：定位 MATH 回退、类别分布和 gate 日志覆盖；没有进行付费因果重合成。
- [x] 方向 A 第二轮零 API 审计：把 265 个回退拆成评分伪影 11 / 无删减 71 / 有删减 183，定位 `_extract_last_number` 的多值答案缺陷（483/5000）；确认零 API 无法判定因果，需冻结候选流的受控实验。
- [x] 方向 A 首轮受控重放（接收端干预，24 题整数子集）：五策略准确率并列 0.875、22/24 逐字相同，为强负向结果；未扩大规模。
- [ ] 方向 A/B/C 的正式方向选择与新费用批准（待用户决策，见文末负向发现条目）。
- [ ] 方向 A 因果阶段：逐条候选纠错验证（需冻结共同候选流），再批准候选生成与 720 次合成的总费用上限。
- [ ] 方向 B/C：完成单方向小规模 smoke 后再定正式成本，不同时展开大规模 sweep。
- [ ] 投稿前：完整文献与数据来源审计、公开可复现入口、官方模板、按新数据重新评估主张。

此前阶段已完成提交；本轮不修改历史 JSON，不假定新方向必定成功，不自动提交或推送新改动。

## 2026-10-23 授权后的探索执行

- 新入口 `experiments/run_frozen_replay.py`：冻结共同完整候选、来源/代码/配置哈希，按题号哈希抽样，不按历史输赢选题。
- 为避免已知评分伪影，首轮限六候选齐全的整数答案任务，严格匹配完整答案；不外推到全部 MATH。
- Full / Random / Cosine / VIB / first-round / direct 六策略，共同候选重放只测接收端；不声称 first-round 已节省候选生成费用。
- 3 题 × 6 策略的 smoke 已完成，18/18 请求有效，合成代理费用 $0.0024486；随后启动 24 题 × 6 策略 × 2 次合成，最多 288 请求。
- 整数池 3031 题，严格完整答案重评分 Full 2628 / VIB 2604；历史评分分别为 2643 / 2618。该敏感性检查不覆盖非标量答案，也不替换主表。
- 请求级原子缓存、失败费用预留、60 秒超时、禁 SDK 自动重试、首次异常即停；成功续跑不重复调用，失败或中断请求需人工审计。
- 修复 rescue 日志身份匹配，不改变实际救援/准入决策；新增回归测试。当前 78 项测试通过。
- 研究新颖性前置核查：ProgRouter (`https://arxiv.org/abs/2608.25992`) 已研究在线质量—成本调度，PEAR (`https://arxiv.org/abs/2606.20621`) 已研究推理期自适应通信。仅“把门控前移”不足以构成明确的新颖性。
- 24 题只是探索，不称 held-out，不用重复合成扩大独立样本量；不据此宣称达到顶会标准或非劣效成立。
- 配置审计中发现 v1 入口误用路由器类默认值，而非论文 harness 的 beta=3.2、lambda=0.08、tau=0.5、gamma=-0.4 和 adaptive 配置。已立即停止，179 个成功请求与 1 个中断预留合计代理费用 $0.0248919；这批及旧 smoke 不用于论文方法结论。
- v2 将完整路由配置写入 manifest 并加测试，分析器拒绝 v1。论文配置下 9/24 题准入历史等同 first-round，其余保留 4–6 条，不是默认配置下的 24/24。
- 重新启动 `results/frozen_replay_pilot_paperconfig_20261023`，仍为 24 题、最多 288 次请求、$3 代理预算上限。固定 random=0.70 / cosine=0.75 未校准成该子集等消息预算，不声称匹配预算选择器优越性。
- 当前 81 项测试通过；论文已将未经验证的 11/71/183 因果归因改成可复核的 MATH/27 格式缺陷与内容混淆说明，主表数字不变，PDF 编译通过且无 Overfull 或未定义引用。

### 论文配置批次结果与负向发现（2026-10-23）

- `results/frozen_replay_pilot_paperconfig_20261023`：24 题 × 6 策略 × 2 次合成 = 288 请求全部有效，代理费用 $0.0389394；`response_model` 全为 pinned `deepseek-v4-flash-0817`，`cached_tokens=0`。
- 五策略（full / random / cosine / vib / first\_round）任务级准确率完全并列 0.875；两两 `delta_pp=0.0`，bootstrap CI `[0,0]`，`tasks_better=tasks_worse=0`。仅 `direct`（不给任何历史）跌到 0.3958，CI `[-66.67,-27.08]`。
- 准入历史确实不同（full 6.0 / vib 4.208 / cosine 4.917 / random 4.375 / first\_round 3.0 条），合成 token 节省 vib 20.99%、first\_round 37.35%、direct 80.66%。
- 逐题复核：**22/24 题在五种历史策略下原始输出逐字相同**；仅 MATH/1629、MATH/2767 输出不同，且两题在全部策略下都判错，不改变准确率。抽查 MATH/1020 的 `prompt_sha256` / `prompt_tokens`（449 / 283 / 81）确认历史内容与长度确实随策略变化，输出仍同为 `答案: -4`。
- 结论（探索性，非独立泛化）：在该整数 MATH 子集上，合成器对「第 4–6 条消息是否进入历史」几乎不敏感，给出首轮 3 条即可；这是对消息级门控价值的强负向信号。
- 边界：只覆盖整数答案、六候选齐全的 24 题；复用同一批历史候选，重复合成不增加独立问题量；`direct` 不是等预算单智能体对照；random `keep_rate=0.70` / cosine `0.75` 未校准为该子集等消息预算。不能据此否证门控在其它任务/规模上的价值。
- 待决策的高收益候选：转向「何时值得调用验证器 / 保留纠错」（自适应验证调度，需与 ProgRouter / PEAR 划清增量）；或在代码任务（纠错对最终代码有实际影响）上重做该受控实验；或先补齐等预算单智能体与跨模型对照。

### 代码评分修正与 HumanEval 重放（2026-10-23）

- 用户选择保留刷新后的原始输出，仅重算可执行评分；先完成修正，再启动已批准的 24 题 × 6 策略 × 2 次重放，代理费用上限 $3。不自动扩展实验。
- 根因：历史提交 `324f3df` 的 synthesis-refresh 评分仅执行 HumanEval 的 `check` 定义而不调用它；`914b6c8` 已修实现，但历史派生 JSON 仍含旧标记。
- `scripts/recompute_code_eval.py` 按 harness 分派重算：HumanEval 显式传 `entry_point`，MBPP 执行顶层断言、不传入口。原始 `results/main_with_math_merged.json` 保持只读；另存 `results/main_with_math_merged.codeeval_fixed.json` 和逐题审计 `results/codeeval_correction_20261023.json`。
- HumanEval：Full 151/164 = 92.07%，VIB 148/164 = 90.24%，Δ−1.83pp，paired bootstrap CI `[-6.71,+3.05]`，McNemar p=0.607239；Full-only=9、VIB-only=6。29 个臂标签由通过转为失败，旧 100%/100% 与零宽区间不再作为证据。
- MBPP：Full 236/257 = 91.83%，VIB 237/257 = 92.22%，Δ+0.39pp，CI `[-1.56,+2.33]`，p=1.0；5 个臂标签由失败转为通过。该差异不是 HumanEval 未调用 check 的同一机制，不能未经验证归因。
- 保留原 ES/MR OR 分组（236/21 题）：ES 改为 Full 99.15% / VIB 98.73%；MR 仍为 9.52% / 19.05%。不重做参考答案早停，也不改变历史分组。
- 已重导表 1、表 2，更新摘要与正文。图表入口同步改用修正产物，并补齐 Pareto 图的 MATH 点与 HumanEval 完整置信区间；不把跨基准散点称为共同 Pareto 前沿。
- 启动前 HumanEval 重放目录只有 dry-run manifest，无付费请求；执行时复用原始候选来源以保持已冻结契约，正确性由可执行测试判定，不读取历史 passed 选题。省去主 harness 的 doctest 自修复，仅研究接收端单次合成，不能直接与主表比较。
- 修正后验证：87 项测试通过；源文件与评分脚本 SHA-256 均匹配溯源记录；逐字段比较确认除通过相关字段及修正溯源块外其余数据完全不变。PDF 编译成功、Overfull hbox 为 0（仍有 Underfull 排版警告）。
- 上述验证通过后，已启动获批 HumanEval 重放，冻结契约哈希 `8b0a6518d73da7e545e60bac60de41fe1ab9df4347b6166b3439df7c0cd0bb84`；请求逐条持久化。未完成全部配对前不报告策略优劣，不启动付费扩展。

### 协作收益边界：零 API 离线审计（2026-10-26）

- 上一阶段按功能提交：`2761f27`（评分与溯源）、`4c88ffa`（代码重放协议与 288 个请求）、`7e087fa`（论文及四基准图表）；提交完成后工作区干净。新阶段不修改原始请求、manifest 或主表数据。
- HumanEval 完成 288/288 有效请求，代理费用 $0.1668918。Full / Random / Cosine / VIB / First-round / Direct 依次为 83.33% / 87.50% / 87.50% / 85.42% / 91.67% / 91.67%；VIB / First-round / Direct 的合成 Token 节省为 3.98% / 11.27% / 21.18%。
- `experiments/analyze_frozen_replay.py` 增加 `collaboration_boundary`：核验请求 ID 与冻结 manifest 指纹一致；每题先平均重复，再对任务聚类 bootstrap；输出逐题双过、双错、单臂通过计数、重复方向相反的情况，以及同提示词评分不一致诊断。
- HumanEval 的 Full−First-round：改善 0 题、变差 2 题、相同 22 题；HumanEval/116、HumanEval/93 在两个重复中均为 Full 失败、First-round 通过。平均差 −8.33pp，探索 CI `[-20.83,0.00]`，不可把区间触及零称为严格显著。
- VIB−First-round：改善 1 题（HumanEval/32，仅一个重复通过）、变差 2 题、相同 21 题，−6.25pp，CI `[-20.83,+4.17]`。本处区间统一按 manifest 任务顺序、固定 seed=20261023 计算；不采用临时诊断中不同任务顺序产生的 bootstrap 端点。随机合成或随机化测试波动不能排除，不把 HumanEval/32 单次改善称为稳定纠错。
- MATH 的 First-round−Direct：改善 14 题、变差 1 题、相同 9 题，+47.92pp，CI `[+27.08,+66.67]`；Full−First-round 的 24 题全部相同。首轮协作与额外一轮的收益必须分开，不能把两者合并成“讨论无用”。
- 可见候选的描述性特征不读取参考答案/隐藏测试，也不执行候选代码：轮次文本长度、第二轮新增词项比例、可解析 Python 围栏及 AST 去重。HumanEval 只有 4/24 题有 Python 代码围栏，MATH 为 0/24；关键差异题也没有围栏，不能据此声称已有可部署的代码证据冲突检测器。词项比例只是正则词项统计，不是语义新颖性或信息量估计。
- 特征只作描述，不在该批任务上训练选择器、拟合阈值或声称预测能力；第二轮特征是生成后才可见，不能用作生成前免费调度信号。两批准确率、费用和任务数不池化；重复合成不增加独立任务量。
- 实际产物：两批目录内 `analysis.json`，以及论文 Appendix 的 receiver-only replay 段落。新增加 6 项测试，全量 93 项通过；本轮无新增 API 请求、无新增代理费用。

### 下一实验的准入协议（未启动付费实验）

1. 问题冻结为“首轮后的额外讨论是否提供可验证净收益”，不继续大规模扫 VIB 超参。上述已见任务只用于发现假设，不用它们证明新策略泛化。
2. 开发与最终评测使用分离任务；仅排除这 24 个题号并不能保证 truly held-out，因为历史主实验已使用整个基准。正式验证需有新任务来源或如实称为新重放子集，并用独立任务族/模型检验迁移。
3. 第一阶段只比较无历史、首轮、完整两轮三个固定接收策略，并单独重放首轮提示词以估计同提示词波动。随机化测试需固定评分种子或独立重评分验证；不把随机测试误差归因于讨论。
4. 若可重复的首轮后正收益存在，再实现只依赖题面、首轮候选及公开证据的规则；禁止把隐藏测试/参考答案用于生成、选择、停止或预算再分配。不能使用已生成的第二轮特征声称节省了其生成费。
5. 若进入生成前调度实验，需计入候选生成、所有重试、验证工具和合成的完整账本；至少包含固定首轮、固定两轮、随机继续及等预算单智能体重采样/修复基线。所有拒绝或截断请求单独归档，不算正确性失败。
6. 主要比较、精度损失容忍范围、样本量设计和费用上限在调用前冻结。先用独立 pilot 估计配对不一致率，再设计统计功效；不以“直到显著”为扩样停止规则。新调用预算须确认，不沿用旧批次剩余的 $3 额度。
7. 退出条件：独立验证仍无额外讨论净收益，或新规则不能优于固定截轮/强单智能体，则不扩展调度器开发，收束为协作收益边界的实证研究；不能仅靠新增模块宣称顶会贡献。

### 新子集边界重放：已批准协议（2026-10-26）

- 用户明确批准两批合计 $3 的新代理预算；HumanEval 和 MATH 各限 $1.50，不转移额度、不自动扩样。仍是新的历史重放子集，不称 held-out 或确认性泛化验证。
- 每批 24 题 × 4 策略 × 2 合成 = 192 请求；策略为 Full、First-round、First-round-control、Direct。主比较冻结为 Full−First-round；相同首轮提示词的独立对照用于估计合成/执行波动，不与首轮池化来人为增加样本量。
- 按 seed=20261026 的 task-ID 哈希抽样，显式排除各数据集旧 pilot 的 24 个题号；manifest 记录排除清单与其 SHA-256，选题不读历史 passed。不训练选择器、不拟合阈值、无需 VIB 新超参。
- 新协议 version=3；HumanEval 的每条输出在两个隔离子进程中用 Python `random.seed(20261026)` 评分，种子不污染父进程，独立执行标签不一致即停。未控制 NumPy、其它 RNG、进程调度或 API 推理随机数，不能声称完全确定性评分；参考答案/隐藏测试仍仅用于评测侧。
- 返回模型必须匹配 pinned `deepseek-v4-flash-0817`，否则记录实际用量并立即中止；失败/中断请求阻止自动续跑。沿用原子请求账本、禁 SDK 重试、60 秒超时和先预留费用再调用的守卫。
- 两批 dry-run 通过，尚无请求时逐项验证旧题零重叠、First-round-control 提示词逐字相同，以及 HumanEval 隐藏测试不进入提示词。按提示词 UTF-8 字节数加余量和最大输出估计的全批代理预留为 HumanEval $0.632047、MATH $0.185882；这是保守预留，不是实际账单或实测新批费用。
- 冻结目录：`results/boundary_humaneval_replication_20261026`（契约 `0bb392e4a856c74bf1493c2f9f042aad172725be07026edde704aa9b635dfd2a`）、`results/boundary_math_replication_20261026`（契约 `eaff7defcfbbb2a03f85002e7b76d4b6b38ee6094f6c759c6aef5f5a603ef693`）。
- 100 项测试通过后开始调用；完整配对前只监控请求有效性和费用，不报告中途准确率。两批独立分析，不合并分母；此固定规模 pilot 不作为精度等价/非劣效检验，不以显著性触发继续扩样。

### 新子集完成与下一步判断（2026-10-26）

- 两批均 `stop_reason=complete`，各 192/192 请求有效、24 个独立任务；重新调用离线分析器与已保存 `analysis.json` 完全一致，请求身份、模型、评分种子与排除清单均通过核验。
- 实际代理费用：HumanEval $0.105123，MATH $0.022278，合计 $0.127401。未新增调用；各自远低于 $1.50 上限，剩余额度不构成自动扩样授权。
- HumanEval：Full 91.67%、First-round 95.83%、First-round-control 93.75%、Direct 95.83%。Full−First-round −4.17pp，探索 CI `[-12.50,0.00]`；改善 0 题、变差 1 题、相同 23 题。首轮合成 Token 节省 11.61%。
- HumanEval/147 在两个重复中 Full 均失败，而首轮、首轮重复对照与 Direct 均通过。第二轮的 Coder 混淆 `i mod 3` 与 `a[i] mod 3`，Full 输出错误闭式计数。题面直接可验证：n=3 时数组为 [1,3,7]，唯一三元组和为11，真值0；Full 的闭式给出1，首轮实现给出0。这是可公开验证的反例，不依赖隐藏测试；但移除哪条消息才能修复仍未做受控干预。
- MATH：Full 95.83%、First-round 91.67%、First-round-control 89.58%、Direct 75.00%。Full−First-round +4.17pp，CI `[0.00,+12.50]`；改善 1 题、变差 0 题、相同 23 题。首轮合成 Token 节省37.85%。因此旧 pilot 中第二轮零收益不能推广到新任务。
- MATH/1791：Full 两次输出16并通过；首轮两次输出20，首轮对照两次输出14，均失败。独立数学核对：变换 u=y−ax、v=y−bx 的雅可比绝对值为 |a−b|，面积为截距差平方除以 |a−b|；两面积之比给出 max(c,d)=3min(c,d)。令 min(c,d)=3k，则 |a−b|=2k²，最小 a+b=2k²+2，c+d=12k，k≥1 时总和最小16。第二轮含正确续算，也含错误质疑，不可仅凭角色或多数一致性标成“有证据”。
- 相同首轮对照在每批48个配对中各有1个标签不一致（2.08%）：HumanEval/38 的一次控制输出漏掉测试所依赖的 encode_cyclic 辅助函数；MATH/4031 的一次控制输出24，而另一输出18。二者提示词哈希相同，说明独立合成仍有波动；这不是估计精确的总体噪声率，也不能直接从主差值中减去2.08pp。
- 主比较仅各有一个差异任务，bootstrap CI 触及零，重复两次不等于独立任务增多。不能把首轮对照池化、合并新旧评分协议或两数据集，来制造显著性或泛化优势。
- 下一步优先零 API 的证据审计：在全部差异案例上分别登记题面可导出的约束、公开可执行反例、正确纠正与错误改写；现有案例作为开发/发现集，不用于证明新选择器。若能形成事前可操作的信号，再设计独立任务与等预算基线下的验证；目前不继续大规模 VIB sweep，不启动新的付费扩展，也不宣称已达到顶会方法贡献。

### 公开证据与反例审计完成（2026-10-26，无新 API）

- 独立入口 `experiments/audit_public_evidence.py` 仅审计 Full−First-round 的全部观测差异/重复方向相反任务，无截断挑选：旧 HumanEval 2 题、旧 MATH 0 题、新 HumanEval 1 题、新 MATH 1 题。四批各生成 `public_evidence_audit.json`，记录 manifest、全部请求内容与审计脚本 SHA-256；原始请求、隐藏测试和 passed 标签不改动。
- 选择发现集可以读取已有 passed 差异，但核验阶段只接收 task_id、公开题面、候选消息和输出；不接收 reference、test_cases、entry_point，也不调用 API。证书由人事后编写，明确不是自动测试生成器、部署选择器或盲测。

| 发现任务 | 公开证据 | Full 两次核验 | 首轮两次核验 | 解释边界 |
|---|---|---|---|---|
| HumanEval/93 | 原题示例 `encode('test')='TGST'` | 均得到 `TgST`，违反公开示例 | 均为 `TGST` | 首轮 Reviewer 已提出错误的原字符映射建议；不能把错误信息首次出现归到第二轮 |
| HumanEval/147 | 原题 n=5 应为1；由数组 [1,3,7] 推导 n=3 应为0 | 均分别得到4和1 | 均分别得到1和0，首轮对照也均满足 | 正确和错误模3映射同处第二轮；仍未证明是哪条消息造成合成错误 |
| HumanEval/116 | 非负整数规则下 `[1,2**32]` 两数 popcount 均为1，按数值排序应保持原序 | 均反序，32位截断违反未设位宽上限的文字规则 | 均保持原序 | 原题多个字面示例与文字排序规则矛盾，且负数示例超出非负描述；不能声称公开例子解决了负数评分语义 |
| MATH/1791 | 雅可比面积公式、整数全局下界和达到下界的见证 (1,3,9,3)，两面积严格为18与72 | 均16，满足独立推导 | 均20，首轮对照均14，均不满足下界最优值 | 首轮 Verifier 已有正确但截断的推导，第二轮补全；不能将效果直接称为新增知识或特定消息的因果贡献 |

- 必须修正先前归因的强度：HumanEval/116 的 32 位错误可用公开非负大整数证伪，但该反例不是对原隐藏负数失败原因的独立证明；MATH/1791 更接近“完成与采纳正确推导”，不是明确的“第二轮纠正首轮整体错误”。
- 新模块以5秒超时的独立子进程执行已有代码输出，异常、超时与反例不混淆；隔离不构成文件系统/网络安全沙箱，不能用于任意不可信外部代码。数学核验不直接执行模型输出。
- 9项独立公开证据测试通过，全量110项测试通过；测试涵盖隐藏材料访问守卫、公开反例、规格矛盾、面积见证、父进程 RNG 不污染、执行异常与未知题保持 unsupported。本阶段费用为0，未新增实验调用。
- 下一阶段应先明确证据抽取是否能在所有任务上运行，以及验证成本与误报率，再谈调度。已有事后证书不足以建立该能力；未知题不会套用当前四个题号的规则。若后续测试生成或候选干预需要 API，则另行冻结协议并确认预算，不使用已见4例作为独立泛化真值。

## 一键导航

[阶段性论文](paper.pdf) · [正文事实源](paper.tex) · [表格生成器](../scripts/export_latex_tables.py) ·
[门控对照数据](../results/gating_sweep/gating_summary.json) · [核心工作流](../src/workflow.py) ·
[代码评分修正](../results/codeeval_correction_20261023.json) · [修正主表数据](../results/main_with_math_merged.codeeval_fixed.json) ·
[HumanEval 边界审计](../results/frozen_replay_humaneval_pilot_20261023/analysis.json) · [MATH 边界审计](../results/frozen_replay_pilot_paperconfig_20261023/analysis.json)

---

## 历史 Markdown 草稿审计（已被上文取代，勿用于当前论文）

以下保留早期审计记录以追溯旧主张。其 +35.93pp 归属、BGE 占位、缺失 trace 后补的推断及建议已过时；当前数字、模型和实验设计以本页上半部分与 `paper.tex` 为准。

> **Purpose.**  本清单列出 paper/paper.md 里**所有定量陈述**所对应的**精确数据来源**（JSON 文件路径 + `summary[ds]` field name + 数值），供：
> 1) 论文写作过程中自查是否「写错一位数」；
> 2) 审稿阶段 rebuttal 一键交叉审计；
> 3) 后续补做 β₀/λ sweep 或 HumanEval digest 后一键知道哪段正文要一起改。
>
> **Regeneration.**  正文来自 `paper/paper.md`。若 JSON 结果重跑：
> ```
> python3 scripts/export_latex_tables.py   # 自动重填 table 1/2/3 .tex
> # 然后手动对照本清单同步 paper.md 正文数字（或 run 专用脚本 TODO: scripts/audit_paper_claims.py）
> ```

---

## §0. Abstract

| Claim (paraphrase) | JSON source | `.summary[ds].field` | Expected value | Paper line |
|---|---|---|---|---|
| Vanilla single-agent GSM8K Pass@1 | `results/vanilla_baseline_3dataset.json` | `gsm8k.full_pass_rate` | **59.29%** | Abstract (1) |
| Full-Comm 3-A GSM8K Pass@1 | `results/3dataset_synthesis_refreshed_split_or.json` | `gsm8k.full_pass_rate` | **95.30%** | Abstract (1) |
| Δ GSM8K Vanilla → Full (pp) | 上面两值相减 (Full-Van) | 派生 | **+35.93 pp** | Abstract (1) |
| HUMANEVAL 3-col (Van/Full/VIB) | vanilla + main | `humaneval.full_pass_rate` ×2 + `main.humaneval.vib_pass_rate` | **93.90 / 100.00 / 100.00** | Abstract (2) |
| MBPP 3-col | vanilla + main | `mbpp.full_pass_rate` ×2 + `main.mbpp.vib_pass_rate` | **89.88 / 90.66 / 91.44** | Abstract (2) |
| GSM8K 3-col (Van/Full/VIB) | vanilla + main | 同上 | **59.29 / 95.30 / 95.22** | Abstract (2) |
| H 消息压缩 % (VIB vs Full) | main | `humaneval.msg_compression_pct` | **+21.14 %** | Abstract (2) |
| G 消息压缩 % | main | `gsm8k.msg_compression_pct` | **+30.49 %** | Abstract (2) |
| H/G token saving % range | main | `.token_saving_pct` ∈ {H 3.88, G 4.96} | **3.9 % – 5.0 %** (文字概括) | Abstract (2) |
| MBPP MR Full pass rate | main | `mbpp.multi_round_full_pass_rate` | **9.52 %** | Abstract (3) |
| MBPP MR VIB pass rate | main | `mbpp.multi_round_vib_pass_rate` | **19.05 %** | Abstract (3) |
| Δ MBPP MR VIB−Full | main | `mbpp.multi_round_pass_delta_pp` | **+9.53 pp / +2/21 solves** | Abstract (3) |
| Two example MBPP task ids | `main.detailed_results[mbpp]` (MR 组 vib_only=1 full_only=0 rows) | `task_id` | **MBPP/260, MBPP/285** | Abstract (3) |

---

## §3.1 Main results / Table 1 paragraph

| Claim | Source | Field | Value | Paper §3.1 line |
|---|---|---|---|---|
| Full → VIB GSM8K Δ pp + 95% CI | `analyze_multi_benchmark.analyze_dataset(ds=gsm8k)` on main | `pass_delta_pp`, `pass_delta_bootstrap_ci_95` | **-0.08 pp, CI [−0.83, +0.68]** | 3.1 bullet 2 |
| HUMANEVAL msg_compress + tok_save | main summary | `humaneval.msg_compression_pct` / `humaneval.token_saving_pct` | **+21.14 / +3.88** | 3.1 bullet 2 |
| GSM8K msg_compress + tok_save | main summary | `gsm8k.msg_compression_pct` / `gsm8k.token_saving_pct` | **+30.49 / +4.96** | 3.1 bullet 2 |
| MBPP overall Δ + 4 vs 2 net VIB-only | main summary | `mbpp.pass_delta_pp`, `mbpp.vib_only_count`/`mbpp.full_only_count` | **+0.78 pp (4 vs 2, McNemar p=0.6875)** | 3.1 bullet 3 |
| MR subgroup size (21 / 8.2%) | main summary | `mbpp.multi_round_count` / `mbpp.multi_round_pct` | **n=21, 8.17% ≈ 8.2%** | 3.1.1 paragraph |
| ES group pass rate VIB=Full | main summary | `mbpp.early_stopped_vib_pass_rate` / `mbpp.early_stopped_full_pass_rate` | **97.88% tie** | 3.1.1 |
| GSM8K Figure 3-a (compress, Δ) pair | main + analyze_dataset | `gsm8k.msg_compression_pct`, pass_delta_pp | **(+30.49, −0.08)** | 3.1.2 |
| HumanEval Figure 3-a (compress, Δ) pair | main + analyze_dataset | `humaneval.msg_compression_pct`, pass_delta_pp | **(+21.14, 0.00, CI [0,0])** | 3.1.2 |
| MBPP Figure 3-a point (slightly neg compress, +Δ) | main | `mbpp.msg_compression_pct`, `mbpp.pass_delta_pp` | **(−0.12, +0.78)** | 3.1.2 |

---

## §3.2 Case Studies

| Claim | Source | Field / join | Value |
|---|---|---|---|
| MBPP/260: task title, function name | `datasets` lib `mbpp sanitized test` via MBPPAdapter | `task_id=260 → entry_point + prompt` | **newman_prime** (N–S–W prime) |
| MBPP/260: "Tester asserts newman_prime(3) == 11 false doctest" — qualitative, claimed failure mode | `trace_results[mbpp][task_id=MBPP/260]` round 2 candidate_messages | need manual run to log Utility score & diff. Mark **[SOFT CLAIM / needs trace snapshot]** until we run `--save-full-trace` | placeholder |
| MBPP/285: entry_point + task | MBPPAdapter | task_id=285 | **text_match_two_three, regex `ab{2,3}`** |
| VIB gating numerical claim (Utility<0) for the misleading doctest on MBPP/260 | not yet serialized — requires `run_multi_benchmark.py` `--save-gate-decisions` flag + rerun | TODO: add CLI arg & write to trace_results; otherwise §3.2 Utility ≈ -0.21 must be marked **[ESTIMATE]** | **[ESTIMATE] ⚠️ flag for rebuttal** |
| "17/21 MR MBPP fail for both Full & VIB" | §Gap1-c script count | MR rows: neither=17 | n=17 | §4 Limitation (3) |
| MBPP/437 "vanilla_only" 类别 | `vanilla_baseline_3dataset.json.detailed_results[mbpp]` + main detailed_results | van full=1 / main full=0 & vib=0 | task_id=MBPP/437 | §4 Limitation (3) |

### ⚠️ §3.2 当前 2 条软声明（审计风险，审稿前必须硬）

1. **Utility ≈ -0.21 for MBPP/260 false doctest.**  当前 `trace_results` schema 不含逐候选的 Utility 数值；§3.2 里引用的 Utility ≈ -0.21 是手算估计，**审稿人会要求 exact**。修复建议：在 `run_multi_benchmark.py` 的 `VIB Sparse 分支` `VIBRouter.evaluate()` 调用后把 `{"task_id","round_idx","role","utility","passed_gate"}` 序列化进 trace_results → 重跑 targeted refresh for 260/285 → 回填 exact 数字。（Gap-follow-up item）
2. **False doctest 文本片段。** 同上：需要把 candidate_messages 原文 128 char 摘要写入 trace；否则故事虽真实但无书面证据。（Gap-follow-up item）

---

## §3.3 Ablation / Table 3 / Figure 3-b

| Claim | Source | Field | Value |
|---|---|---|---|
| GSM8K ablation frozen_only: N / Full / VIB / Δpp | `results/gsm8k_ablation_frozen_only.json` summary.gsm8k | `total_tasks / full_pass_rate / vib_pass_rate / pass_delta_pp` | **200 / 97.0 / 96.5 / −0.50** | Table 3 row 1, Figure 3-b 左 |
| GSM8K ablation frozen_only msg_compress tok_save | frozen_only json | `msg_compression_pct / token_saving_pct` | **+30.25 %, +3.91 %** | |
| GSM8K ablation frozen_only Δ 95% CI | `analyze_dataset(frozen_only.detailed_results.gsm8k, ...)` | pass_delta_bootstrap_ci_95 | **[−1.50, +0.00] pp** (reported in §3.3 text "CI [-1.50,0.00]") | §3.3 text |
| GSM8K ablation +role weights: N / Full / VIB / Δpp | `results/gsm8k_ablation_frozen_plus_role.json` summary.gsm8k | same 4-tuple | **200 / 96.0 / 96.0 / 0.00** | Table 3 row 2, Figure 3-b 右 |
| +role weights msg_compress + tok_save | role json | `msg_compression_pct / token_saving_pct` | **+25.17 %, +4.17 %** | §3.3 |
| "role-weights compresses less but closes accuracy gap" 定性概括 | Δpp 0 (from -0.5) + compress 25.17 < 30.25 | 派生比较 | 文字描述 OK | §3.3 |

---

## §2 Method 参数 defaults（超参声明审计）

| Symbol | Meaning | Claimed default | Source location |
|---|---|---|---|
| $\beta_0$ | initial VIB multiplier | **2.0** | 见 `src/vib_router.py` `VIBSparseCommRouter.__init__` 默认签名 |
| $\lambda$ | exponential decay over rounds | **0.15** | 同上 `lambda_decay` 参数 |
| $\tau$ | InfoNCE temperature | **0.2** | 同上 `temp_tau` |
| $\gamma$ | gate acceptance threshold | **0.0** | 同上 `gamma` |
| Embedding model $E(\cdot)$ | sentence embedder used for cos MI estimate | **BGE-small-v1.5** (placeholder text) | paper.md §2.2 "BGE-small-v1.5"  ⚠️ **VERIFY IN vib_router.py before submission — currently mocked with TF-IDF or local sim?** |

> **⚠️ CRITICAL AUDIT GAP (pre-submission blocker):**  §2.2 声称使用 "BGE-small-v1.5 sentence embeddings"；需在 `src/vib_router.py` 里核对真实实现（sentence-transformers / DashScope / TF-IDF fallback?）。**如果实际上走的是轻量级 fallback（TF-IDF / Jaccard / subword），则 §2.2 文字不能写 BGE-small** —— 必须如实报告实现，否则审稿人跑 `tests/test_vib_router.py` 直接发现 mismatch，属于可拒稿级错误。（Gap-high priority）

---

## §1 Contributions 3 条 (C1/C2/C3) vs 实验覆盖矩阵

| Contribution 标签 | 正文支持的实验表格/图 |
|---|---|
| C1: embedding-only gate, no extra LLM call | §2.2 formula + 代码 (`vib_router.py` 无 LLM 依赖). 开销可用 `scripts/benchmark_router_latency.py` (TODO) 做 ms/次 |
| C2: 3-col unified protocol + 95% CIs | **Table 1** (Wilson + Boot) + Figure 3-a (Boot CI whisker) + analyze_multi_benchmark.py CI 公开 |
| C3: tail case F1 mechanism | §3.2 two case studies + Table 2 MR row.  ⚠️ 需要补 Utility trace (见软声明) |

---

## Post-write todo (从草稿反推的补实验/补日志项)

1. **[HIGH] VIB Router 真实 embedding 栈核对** — 决定 §2.2 怎么写。
2. **[HIGH] Utility trace serialization（+ rerun 260/285 targeted）** — 让 §3.2 两条软声明变硬（Gap-audit soft-claim 1/2）。
3. **[MED] Router latency benchmark (ms / msg)** — 支撑 C1 的 "millisecond-level" 文字。
4. **[MED] MBPP/437 van-only failure 定性笔记** — 用于 §4 Limitation 第 (3) 条 "不是通信瓶颈而是 synthesis 瓶颈" 的举证，可从 vanilla_baseline trace 里抽 Coder 最终输出 vs 标准答案做 1 段 100 字摘要。
5. **[LOW] β₀ / λ 2D sweep (e.g. β₀ ∈ {1,2,4}, λ ∈ {0.05,0.15,0.3})** — 把 Table 3 升级成 3×3 热图；如果时间紧（顶会 deadline 前 3 天）可以不跑，当前 frozen/role 两列足够。
6. **[LOW] 正文引用 → bibtex key 映射清单**：§5 中提到的 Du2023AgentCoder, Alemi2017VIB, Tishby2015IB, Razeghi2022InfoBERT, Xia2024DyLAN 等 10 篇需要和 `paper/refs.bib` 一一对应（Gap-appendix refs）。

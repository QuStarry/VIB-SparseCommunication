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
71 题（26.8%）两臂广播消息数相同（Δ=0），门控无法解释失败，属采样噪声；
183 题（69.1%）VIB 广播数更少，才是需要因果实验验证的部分。
反方向也有 6 个 VIB-only 题属评分伪影。机制来源是 `MathEvaluator._extract_last_number`：
正则 `[-+]?\d[\d,]*` 对含逗号的多值答案（483/5000≈9.7%）会把 `12, 10, 6` 与 `12,10,6`
解析成 `6` 与 `12106`，同一答案因空格差异判定相反。
据此零 API 可排除约 31%（11+71）的回退与门控压缩有关，但不能证明「关键纠错误删」：
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

目前缺私有网关输入/输出及缓存命中单价，不能给可信人民币/美元金额。
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
- [ ] 方向 A 因果阶段：逐条候选纠错验证（需冻结共同候选流），再批准候选生成与 720 次合成的总费用上限。
- [ ] 方向 B/C：完成单方向小规模 smoke 后再定正式成本，不同时展开大规模 sweep。
- [ ] 投稿前：完整文献与数据来源审计、公开可复现入口、官方模板、按新数据重新评估主张。

本次没有修改历史 JSON，没有假定新方向必定成功，没有提交或推送新改动。

## 一键导航

[阶段性论文](paper.pdf) · [正文事实源](paper.tex) · [表格生成器](../scripts/export_latex_tables.py) ·
[门控对照数据](../results/gating_sweep/gating_summary.json) · [核心工作流](../src/workflow.py)

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

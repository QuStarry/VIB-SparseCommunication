# 生成前调度实验协议（预注册）

作者：屈占祥（Zhanxiang Qu），独立研究者。ORCID：0009-0000-0137-9751。

本文件在**任何真实模型调用之前**冻结协议、规模与预算口径。未获授权前不发起调用。若任何字段在开跑后被修改，须记录修改原因并视为探索性结果。

## 1. 背景与动机

当前门控位于候选生成**之后**：`src/workflow.py` 的 `_agents_node` 先让全部角色生成候选，`_vib_gatekeeper_node` 再逐条门控。因此候选生成费用在门控之前已经发生、属于沉没成本，门控只能省下"后续合成输入"，无法省下生成本身；这解释了"消息数降 21%–31%、输入词元仅降 4%–5%、精度不动"的结构性天花板。

生成前调度把决策**前移到候选生成之前**：被跳过的角色完全不调用模型，其成本为零。这是唯一能同时降低总 Token 且不以精度为代价的结构性路径。

## 2. 待检验假设

- **H1（成本）**：相对 Full 通信，Coverage 调度在端到端总 Token（含全部候选生成与最终合成）上降低 **≥15%**。
- **H2（精度非劣）**：Coverage 相对 Full 的任务准确率差 **≥ −1pp**（单侧非劣界 −1pp）。
- 探索性（不作确认性声明）：Coverage 相对 RandomSkip 的准确率差，用于隔离"单纯少调用"与"任务证据调度"。

## 3. 固定因素（单因素设计）

- 团队：HumanEval/MBPP = Architect/Coder/Reviewer；数学 = Planner/Solver/Verifier。
- `max_steps=2`；温度 0；输出上限沿用主基准；相同任务集、相同题面、相同评分器。
- 关闭数学角色权重与推理保护（沿用主配置）：`enable_math_role_weights=False`、`enable_reasoning_guard=False`。
- **证据约束**：所有调度器只能使用决策时可见的证据（goal、已批准历史、轮次 `step_t`、角色名），**不得**使用参考答案或隐藏测试，不得据此早停、选优或补生成。

## 4. 处理条件（5 个）

| 条件 | 机制 | 生成前是否决策 | 说明 |
|---|---|---|---|
| Full | 全部角色运行，无条件放行（`BypassVIBRouter`） | 否 | 上界参考 |
| PostGate | 全部角色运行，VIB-Sparse 生成后门控 | 否 | 现状方法 |
| FixedTrim | 第 1 轮全部角色，之后不再运行 | 是（固定） | 平凡裁剪基线 |
| RandomSkip | 与 Coverage 同数量，但随机决定保留哪些角色（逐题逐轮固定种子） | 是（无任务证据） | 数量匹配对照 |
| Coverage | 第 1 轮全部角色；之后仅运行历史未出现的角色，并强制含收口角色 | 是（历史覆盖度） | 新调度器 |

实现：`src/pre_generation_scheduling.py`（`role_coverage_scheduler` / `count_matched_random_scheduler` / `fixed_round_scheduler`）经 `VIBMultiAgentWorkflow(pre_generation_scheduler=...)` 挂接。

RandomSkip 是 Coverage 的**数量匹配随机对照**：复用 Coverage 决定「本轮保留几个角色」的规则，但保留哪些角色由逐题逐轮的固定种子均匀随机决定（`count_matched_random_scheduler`）。两臂的调用削减量因此逐轮对齐，精度差异可归因于"调度的角色选择"而非"调用次数"。

## 5. 主比较与统计口径

- 主比较：**Coverage − Full**（总 Token 与任务准确率）。
- 关键对照：**Coverage − RandomSkip**（隔离调度信号）。
- 统计单位：任务；配对 bootstrap 10000 次；单比例用 Wilson 区间；沿用多重检验（Bonferroni）说明。
- 报告三口径分离：广播消息数、输入词元、完整费用，避免以消息数当通信收益。

## 6. 规模与预算（待授权）

历史实验已耗尽 HumanEval（164/164）、MATH（5000/5000）、MBPP `test`（257/257）、GSM8K `test`（1319/1319），无新鲜余量。故改用**与历史划分完全不重叠**的新鲜题池（本地 HF 缓存可离线复现）：

| 角色 | 数据集 / 划分 | 题数 | 新鲜性依据 |
|---|---|---:|---|
| 主集 | GSM8K `train`（`GSM8K-train/0..299`） | 300 | 历史仅用 `test`（`GSM8K/0..1318`），命名空间不重叠 |
| 副集 | MBPP `sanitized` `train`+`validation` | 163 | 全部（train 602–809 = 120；validation 554–600 = 43）；历史仅用 `test`（id 11–479），无交集 |

合计 463 题；两池分两次独立运行，各自独立产物与指纹。选题规则：历史题排除后按 `task_id` **数值升序**取前 N，以保证上表题号区间可复现。

- 每变体至多 7 次请求/题（3 角色 × 2 轮候选 + 1 次合成）；5 个变体。
- 预估请求量 ≤ 5 × 463 × 7 ≈ **16,205 次**（被跳过的角色不发起请求，实际更少）。
- Token 锚点：GSM8K `test` 全量 full 臂 3.09M / 1319 ≈ **2.34k Token/题·变体**；MATH `test` ≈ 2.94k。估计取 2.4k（数学）/ 3.0k（代码）。
- 预估 Token ≈ 5 × (300 × 2.4k + 163 × 3.0k) ≈ **6.05M**；按历史混合单价（输入 $0.30/M、输出 $1.20/M，约 7:3 配比 ≈ $0.57/M）估算 ≈ **$3.4（≈25 元）**。
- 上限：单批 **≤100 元**。实际账单以 ledger 记录为准；unknown/pending usage 不计为零成本。

### 6.1 模型与密钥冻结（实跑前实测确定）

- 模型：**`deepseek-v4-flash`**。原计划的 `deepseek-chat` 在私有网关返回 404 `model_not_found`（不在模型目录内），已弃用。
- 密钥：`--provider private`，环境变量 `VIB_PRIVATE_API_KEY_ENV=PRIVATE_API_KEY`。原钉死的 `PRIVATE_API_KEY_1` 实测返回 429 `apikey_quota_exhausted`，已改用 `PRIVATE_API_KEY`。
- **跨批不可比声明**：本批使用 `deepseek-v4-flash`，与历史实验的 `deepseek-chat` 结果**不可直接跨批比较**（模型、网关、单价均不同）；所有 H1/H2 判读只在**本批内部的 5 个条件之间**做配对比较。
- 运行命令（须获授权后执行）：

```
set -a; source configs/pro.env; set +a
VIB_PRIVATE_API_KEY_ENV=PRIVATE_API_KEY \
python3 experiments/run_pre_generation_scheduling.py \
  --datasets gsm8k_train --limit 300 --provider private --model deepseek-v4-flash \
  --exclude-manifest <历史结果 JSON ...> \
  --output-dir results/pre_generation_scheduling_<YYYYMMDD>/gsm8k_train

VIB_PRIVATE_API_KEY_ENV=PRIVATE_API_KEY \
python3 experiments/run_pre_generation_scheduling.py \
  --datasets mbpp_train --limit -1 --provider private --model deepseek-v4-flash \
  --exclude-manifest <历史结果 JSON ...> \
  --output-dir results/pre_generation_scheduling_<YYYYMMDD>/mbpp_train
```

注：`--exclude-manifest` 在此处不期望过滤任何题（两池 id 与历史无交集）；若题数异常减少，即说明「新鲜」主张被破坏，须中止排查。GSM8K `train` 余量充足（7473 题），如需更高功效可在同预算内上调 `--limit`。

## 7. 产物与不变量

- 新产物目录 `results/pre_generation_scheduling_<YYYYMMDD>/`，独立指纹，**不修改**历史 JSON 的 history/output/passed/Token/费用字段。
- 每次请求写 ledger，保留 usage 与 request_context；`--exclude-manifest` 强制排除历史题。
- 隐藏测试与参考实现仅用于评分，不用于抽题、提示、选优、补生成或早停。
- 无正结果时如实报告，不早停选取更优子集。

## 8. 判读规则（③ 预注册）

- 若 H1 与 H2 同时成立：可作为"生成前调度"的探索性正证据，仍须注明单团队、两轮、单种子限制。
- 若仅 H1 成立（省 Token 但精度越过非劣界）：报告为成本--精度权衡，不声称等效。
- 若二者皆不成立：如实报告阴性结果，不调整协议后重跑同一主比较。
- 所有结果标注为探索性，除非达到预声明的规模与多种子门槛。

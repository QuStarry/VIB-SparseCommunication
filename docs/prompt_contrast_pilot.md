# 完整代码提示词配对小试验

本工具执行探索性的历史共同候选流对照，不是VIB/Full因果实验，也不是严格预算driver的替代品。原始164题准备、历史结果、冻结的评分bundle及核心路由不修改。

## 冻结协议

| 项目 | 规则 |
|---|---|
| 抽样 | 排序164个任务ID，以seed20261028不放回抽24题 |
| 对照 | 历史简洁、当前仲裁、规格优先，各题各两次 |
| 候选流 | 同题三组共用同一历史Full六候选 |
| 请求 | 全部144个job先声明，顺序seed20261029，串行单次尝试 |
| 参数 | temperature=0，max_tokens=1024，SDK timeout60/retries0 |
| 凭证 | 仅`PRIVATE_API_KEY_1`，必须显式选择，不回退 |
| 预算 | 历史代理输入$0.30/百万、输出$1.20/百万，观察停机$1 |
| 截断 | 保留原文与真实usage，计失败，不补生成 |
| 未知 | pending/unknown/quota/模型漂移/缺账目即停止，不重发 |
| 评分 | 全部生成成功结束后载入隐藏测试，每份完整输出两个独立进程 |
| 统计 | 每题先取两次结果均值，10000次任务级配对bootstrap |

字节数加1024是规划场景，不是已验证prompt Token上界。代理费用不是网关实际账单，不能据此保证美元或人民币invoice上限。提示长度不同，本实验不是等实际费用对照。

## 执行

先运行测试：

```bash
python3 -m pytest tests/test_prompt_contrast_pilot.py tests/test_prompt_contrast_pilot_analysis.py -q
```

只冻结、不请求：

```bash
python3 -m experiments.run_prompt_contrast_pilot \
  --output-dir results/prompt_contrast_pilot_20261028
```

真实执行须具有用户明确的小额实验授权：

```bash
VIB_PRIVATE_API_KEY_ENV=PRIVATE_API_KEY_1 \
python3 -m experiments.run_prompt_contrast_pilot \
  --output-dir results/prompt_contrast_pilot_20261028 --execute
```

本地分析，无API调用：

```bash
python3 -m experiments.analyze_prompt_contrast_pilot \
  --output-dir results/prompt_contrast_pilot_20261028
```

## 持久化与恢复

发送前账目pending必须fsync。SDK返回后先将代码原文、响应身份和内容hash保存至`responses/`，再让LLMClient追加usage终态。截断原文也保存，不能因为client拒绝截断而丢失已付费证据。

同目录再次运行先核对manifest、源码、输入、账目与所有响应hash。完整报告复用，不初始化客户端；没有报告但已有合法终态与原文时只继续从未发送的job。pending、未知用量或孤立原文不自动重发。用量缺失时总费用为未知，仅另列已知用量小计。

`.run.lock`防止并发执行；异常中断残留锁不能自动删除。应先确认原进程不再运行并核验账目，不通过新目录绕过未决请求。源码或输入变化使旧协议失效，不能借此复制付费任务并称作正常恢复。

本地评分设施错误不触发模型请求；未完整生成144个job时不生成配对准确率，避免预算或失败截出的子集被误报为完整实验。两次评分不一致会报设施错误，不投票。隔离解释器不继承凭证，但不是网络或文件系统安全沙箱。

## 解释边界

这24题来自已审计的历史存档，是探索性样本，不能作为新的独立确认集。重复生成不是额外独立任务；成功比例不是best-of-two。三个配对区间未校正多重比较，跨零不等于等效，不能据小样本证明新方法显著优越。

## 48题不重叠复验

首轮源码绑定manifest和结果hash，保持不变。第二轮使用独立入口，从首轮未抽中的140题按seed20261030抽48题，三提示各两次，共288请求，顺序seed20261031。原消息hash、候选流、温度、输出cap、串行尝试和第二密钥不变。

```bash
python3 -m experiments.run_prompt_contrast_replication \
  --output-dir results/prompt_contrast_replication_20261028
```

本轮用户已授权继续小额实际实验，执行命令：

```bash
VIB_PRIVATE_API_KEY_ENV=PRIVATE_API_KEY_1 \
python3 -m experiments.run_prompt_contrast_replication \
  --output-dir results/prompt_contrast_replication_20261028 --execute
```

全部生成结束后本地评分：

```bash
python3 -m experiments.analyze_prompt_contrast_replication \
  --output-dir results/prompt_contrast_replication_20261028
```

主比较预先固定为历史简洁减当前仲裁，任务级10000次bootstrap和非平局任务均值方向的双侧精确符号检验（alpha0.05）；另外两对为探索性未校正比较。均值与方向不是同一个统计量，不能选择有利结果；首轮24题与第二轮48题分开报告，不合并成确认性72题结果。

不重叠只表示与首轮任务ID无交集，这些题仍来自此前已审计的同一HumanEval存档，不是全新独立数据集，不排除预训练污染，也不证明VIB门控因果收益。

复验入口沿用原文与账目核验帮助函数，独立的`src/exploratory_prompt_session.py`负责冻结会话执行。未知用量、部分收到响应的恢复和完整缓存复用规则与首轮一致；没有报告但存在missing-evidence停机信息时费用仍未知，不能因已有其他请求的known usage而误报总费用完整。

## 一键导航

| 内容 | 入口 |
|---|---|
| 生成入口 | [run_prompt_contrast_pilot.py](../experiments/run_prompt_contrast_pilot.py) |
| 评分与统计 | [analyze_prompt_contrast_pilot.py](../experiments/analyze_prompt_contrast_pilot.py) |
| 严格预算driver | [experiment_session_driver.md](experiment_session_driver.md) |
| 当前事实源 | [gap1_claim_audit_trail.md](../paper/gap1_claim_audit_trail.md) |
| 本轮结果 | [README.md](../results/prompt_contrast_pilot_20261028/README.md) |
| 不重叠复验入口 | [run_prompt_contrast_replication.py](../experiments/run_prompt_contrast_replication.py) |
| 复验评分 | [analyze_prompt_contrast_replication.py](../experiments/analyze_prompt_contrast_replication.py) |
| 48题复验结果 | [复验README](../results/prompt_contrast_replication_20261028/README.md) |

# Multi-Benchmark Analysis

Source: `results/multi_benchmark_limit10_workers4.json`

## 总体判断

- 仅看 `Pass@1` 百分比会放大结论，需要结合配对胜负数、显著性和逐题差分一起看。
- `full_only_count` 表示 `Full 成功但 VIB 失败`；`vib_only_count` 表示 `VIB 成功但 Full 失败`。
- `mcnemar_pvalue` 越小，说明两种设置在同一批任务上的差异越不像随机波动。

## 数据集逐项分析

### humaneval

- `Pass@1` 变化: `+10.00 pp`
- 配对胜负: `VIB only = 1`，`Full only = 0`，净增益 `+1` 题
- 配对显著性: `McNemar p = 1.0`
- 双方都成功: `9`，双方都失败: `0`
- 消息不变占比: `20.00%`，平均每题少发消息: `1.7`
- 平均每题少用 Token: `157.3`
- Token 分解: `prompt saving = 7.85%`，`completion saving = -3.19%`
- `VIB` 优势样例: `HumanEval/1`

### mbpp

- `Pass@1` 变化: `+10.00 pp`
- 配对胜负: `VIB only = 1`，`Full only = 0`，净增益 `+1` 题
- 配对显著性: `McNemar p = 1.0`
- 双方都成功: `9`，双方都失败: `0`
- 消息不变占比: `90.00%`，平均每题少发消息: `0.1`
- 平均每题少用 Token: `62.7`
- Token 分解: `prompt saving = 1.74%`，`completion saving = 5.57%`
- `VIB` 优势样例: `MBPP/56`

### gsm8k

- `Pass@1` 变化: `+0.00 pp`
- 配对胜负: `VIB only = 0`，`Full only = 0`，净增益 `+0` 题
- 配对显著性: `McNemar p = 1.0`
- 双方都成功: `10`，双方都失败: `0`
- 消息不变占比: `0.00%`，平均每题少发消息: `3.2`
- 平均每题少用 Token: `310.3`
- Token 分解: `prompt saving = 16.76%`，`completion saving = -1.17%`

## 对先前讨论的修正

- HumanEval 和 MBPP 的提升需要谨慎表述。如果净增益只有 1 题，就更适合写成“结果持平、略优于 Full”，而不是“显著提升”。
- GSM8K 的下降也不能只看 `-0.99pp`，还要看它对应的是多少条配对逆转样本，以及这些样本是否集中在某些推理模式上。
- “消息压缩率远高于 Token 节省率”这个判断方向通常是对的，但需要 `prompt/completion` 拆分数据支撑，不能只凭经验解释。
- MBPP 压缩率低不一定只是“代码任务信息密度高”，还可能与当前实验配置有关，例如 `max_steps=2`、三智能体上限接近固定 6 条消息、以及消息模板较短。

## 建议的下一步

- 优先分析 `full_only`/`vib_only` 的逐题模式，尤其是 GSM8K 的 `full_only` 逆转样本。
- 后续实验固定记录 `prompt/completion` 拆分，避免再对 Token 节省原因做不可证伪的推断。
- 若要引入更强的自适应机制，先验证当前客户端/模型是否真的支持 `logprobs`、低开销前缀探针和可复用 KV cache；否则相关讨论仍属于架构设想，而不是可直接落地的实验结论。

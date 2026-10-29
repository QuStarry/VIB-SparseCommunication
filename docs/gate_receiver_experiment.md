# 共同候选门控接收端实验

这个实验隔离接收历史的筛选作用：所有组使用同一题面、同一六候选Full历史、同一规格优先提示，只有给合成器看的历史不同。候选是既有存档，第二轮没有按新筛选重新生成，不能据此宣称完整多智能体协作因果优势。

## 冻结协议

从首轮24题与复验48题之外抽32题，seed20261101。Full、VIB、随机消息数匹配、余弦消息数匹配各题两次，共256次，随机顺序seed20261102。temperature0、max_tokens1024、串行single-attempt、SDK retries0/timeout60。第二私有密钥须显式选择，无fallback。

本地MiniLM CPU/eval，384维、L2归一化，snapshot与文件hash冻结。实际encoder max_seq_length256，长代码讨论截断，不是完整代码语义互信息。候选正文与带角色历史分别编码，增量历史逐条评估，匹配HumanEval主流程，保留全部现有路由启发式，无新摘要/最小角色数/测试反馈。

随机在每轮三条候选中不放回抽k条，余弦贪心选择与已选历史最大相似度最低的k条；k为该题该轮VIB实际通过数。两种选择都保留原候选输出顺序。数量匹配不是Token、角色构成或实际费用匹配，也不是核心Bernoulli random/threshold topk实现。

每题随机选择只有一个预声明task-salted种子，两次合成共享同一冻结子集。重复次数衡量合成波动，不是随机选择策略的多种子评估；总体推断须考虑这一限制。角色组成也未匹配，因此VIB与随机差异包含角色选择与既有启发式，不是单独utility公式的贡献。

## 执行与恢复

离线设计，无模型API：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m experiments.prepare_gate_receiver \
  --output-dir results/gate_receiver_design_20261028
```

只冻结模型请求：

```bash
python3 -m experiments.run_gate_receiver \
  --design results/gate_receiver_design_20261028/design.json \
  --output-dir results/gate_receiver_20261028
```

本轮用户已授权继续小额实际测试，付费入口：

```bash
VIB_PRIVATE_API_KEY_ENV=PRIVATE_API_KEY_1 \
python3 -m experiments.run_gate_receiver \
  --design results/gate_receiver_design_20261028/design.json \
  --output-dir results/gate_receiver_20261028 --execute
```

完整生成后隔离评分，无模型API：

```bash
python3 -m experiments.analyze_gate_receiver \
  --output-dir results/gate_receiver_20261028
```

每次发送前pending账目fsync，收到响应先保存原文再追加usage终态；未知usage、未决请求、孤立原文、429或模型漂移停机，截断保留usage且计失败、不重试。完整缓存复用不初始化client；已收到且账目结算的任务只恢复本地或继续从未发送的job。设计、源码或原文hash变化拒绝沿用旧协议。

同目录互斥锁防双开。未完成全部256请求不发布完整质量比较，不能把失败/预算截出的子集当全实验。评分设施错误不触发重新生成，两次独立解释器评分不一致会停止，不投票。子进程最小环境不继承凭证，但不是网络/文件系统安全沙箱。

## 指标解释

主比较预先指定VIB减随机消息数匹配，32题重复均值后10000次任务级bootstrap，seed20261101；非平局题方向做双侧精确符号检验，alpha0.05。其他比较为探索性未校正；两个统计量不同，跨零不是等效，没有非劣margin。

消息与历史字节按每题唯一历史计，不随两次重复翻倍。费用是本轮新合成输入输出usage，按历史代理输入$0.30/百万、输出$1.20/百万计；$1是观察停机阈值，不是网关invoice硬保证。字节加1024是规划场景，不是verified Token上界。

所有组相同的是每题请求数、输出cap和候选来源；实际input、output、费用不同。候选生成开销已经发生且不在本轮新账目，不称端到端费用节省。VIB保留现有角色、code-diff、fix关键词等复合启发式，因此对随机的差异也不能归因于单独VIB公式。

任务与前两轮不重叠，但都来自已审计HumanEval，不能声称全新独立确认数据或排除预训练污染。模型ID匹配只是返回声明，不是官方immutable snapshot认证。核心路由与论文主表不因单轮小样本自动更新。

## 一键导航

| 内容 | 入口 |
|---|---|
| 离线设计 | [prepare_gate_receiver.py](../experiments/prepare_gate_receiver.py) |
| 请求入口 | [run_gate_receiver.py](../experiments/run_gate_receiver.py) |
| 评分分析 | [analyze_gate_receiver.py](../experiments/analyze_gate_receiver.py) |
| 真实编码与投影结果 | [设计README](../results/gate_receiver_design_20261028/README.md) |
| 合成真实结果 | [实验README](../results/gate_receiver_20261028/README.md) |
| 单一事实源 | [审计文档](../paper/gap1_claim_audit_trail.md) |

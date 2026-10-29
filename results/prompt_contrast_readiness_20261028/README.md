# Prompt 对照执行前就绪报告

完整请求数：984。模型：`deepseek-v4-flash-0817`。凭证仅记录变量名`PRIVATE_API_KEY_1`与是否存在，不导出或哈希其值。端点仅完成本地配置解析，未认证或验证在线模型身份。

## 条件费用场景

历史2026-10-23代理价：输入$0.30/百万、输出$1.20/百万。全部输出上限按该历史价计为$1.2091392，不代表实际会用满输出上限。

以“UTF-8消息内容字节数加每请求1024假设开销”作输入单位的规划场景合计$2.6758938。这不是实测Token、已证明Token上界、当前网关账单或预算合规保证，不得用于填入reviewed_provider_bounds。

## 执行状态

候选清单就绪：False。付费授权：False。新模型调用：0。

阻塞项：current_price_evidence_missing, provider_prompt_bounds_missing, manifest_budget_not_approved。

无完整证据时不输出candidate_manifest.json；有候选清单也必须另行取得绑定其SHA256与预算的批准。历史小规模代理计价授权不自动批准本次984请求。

后续需私有网关实际价目/缓存计费依据、model对应tokenizer和chat template的上界说明，以及对新manifest的预算确认。公开模型价格或少量观测usage不能证明全部请求的真实上界。

## 一键导航

| 内容 | 入口 |
|---|---|
| 端点、分组场景与就绪详情 | [readiness_report.json](readiness_report.json) |
| 已冻结真实候选流 | [准备报告](../prompt_contrast_preparation_20261028/README.md) |
| 隔离调度与证据契约 | [使用说明](../../docs/experiment_session_driver.md) |

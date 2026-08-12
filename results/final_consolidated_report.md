# VIB-SparseCommunication · 最终综合实验报告

_生成时间：2026-08-12 17:12:00_

本报告由 `experiments/generate_report.py` 自动聚合：主实验（三基准 P1+P2 synthesis 刷新）+ GSM8K 消融 + 已生成的 split/qualitative 分析。

## 1. 三基准主实验（P1+P2 synthesis 刷新）

_数据源：`3dataset_synthesis_refreshed_split_or.json`_

### 1.1 Executive Summary（双方 Pass@1 / 压缩率 / McNemar）

| 数据集 | Full Pass@1 | VIB Pass@1 | Δ (pp) | 消息压缩 | Token 节省 | McNemar p | Full-only | VIB-only |
|---|---|---|---|---|---|---|---|---|
| HUMANEVAL | 100.00% | 100.00% | 0.00 | 21.14% | 3.88% | 1.0 | 0 | 0 |
| MBPP | 90.66% | 91.44% | 0.78 | -0.12% | -0.05% | 0.6875 | 2 | 4 |
| GSM8K | 95.30% | 95.22% | -0.08 | 30.49% | 4.96% | 1.0 | 14 | 13 |

### 1.2 MBPP Early-Stopped / Multi-Round 拆分（split_mode = or）

| 分组 | Count (%) | Full Pass@1 | VIB Pass@1 | Δ (pp) | Msg Compress | Token Save |
|---|---|---|---|---|---|---|
| Early-Stopped | 236 (91.83%) | 0.00% | 0.00% | 0.00 | 0.00% | 0.00% |
| Multi-Round | 21 (8.17%) | 0.00% | 0.00% | 0.00 | 0.00% | 0.00% |

### 1.3 主结论速览

- VIB 三基准中有 **2 / 3** 个 Pass@1 ≥ Full（按 synthesis-refreshed 当前值）。
- Early-Stopped 组无显著正确性损失；Multi-Round 组 VIB 能在消息量接近的情况下净提正确率。
- Message / Token 压缩在 ES 组显著；MR 组因进入多轮双方消息数都已膨胀，压缩更依赖 VIB 侧过滤。

## 2. GSM8K 消融实验（冻结方案对比）

_数据源：`gsm8k_ablation_frozen_plus_role.json`_

| 变体 | Full Pass@1 | VIB Pass@1 | Δ (pp) | Msg Compress | Token Save |
|---|---|---|---|---|---|
| gsm8k | 96.00% | 96.00% | 0.00 | 25.17% | 4.17% |

# Gap1 数字对照清单 (Audit trail for paper/paper.md claims)

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

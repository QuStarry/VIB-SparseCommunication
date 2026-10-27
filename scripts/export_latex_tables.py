r"""
Phase 3-a: 把 3 类实验 JSON 结果自动导出为 LaTeX 顶会论文表格。

Inputs (默认路径，CLI 可覆盖):
  --vanilla     results/vanilla_with_math_merged.matheval_fixed.json
  --main        results/main_with_math_merged.codeeval_fixed.matheval_fixed.json
                                                          (含 MATH；由 scripts/recompute_code_eval.py
                                                           重算代码基准通过标记后的修正产物)
  --ablfrozen   results/gsm8k_ablation_frozen_only.json
  --ablrole     results/gsm8k_ablation_frozen_plus_role.json
  --outdir      paper/tables/

Outputs:
  paper/tables/table1_main_three_methods.tex       (Table 1: Vanilla / Full / VIB × 3 datasets,
                                                      with Δpp 95% Boot CI + Wilson CI + McNemar)
  paper/tables/table2_mbpp_es_mr_split.tex          (Table 2: MBPP Early-Stopped vs Multi-Round, 3 methods)
  paper/tables/table3_gsm8k_ablation.tex            (Table 3: GSM8K ablation: frozen_only vs
                                                      frozen+role_weights)
  paper/tables/table5_crossmodel.tex                (Table 5: cross-family GSM8K N=200 transfer)
  paper/tables/_README.txt                           (how to \input in manuscript)

Also prints a plain-text preview of the tables.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
import analyze_multi_benchmark as an


# ---------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------

@dataclass
class DatasetRow:
    ds: str
    label: str
    total: int
    vanilla_pass: float
    full_pass: float
    vib_pass: float
    delta_vib_full: float
    delta_vib_full_ci: Tuple[float, float]
    delta_vib_vanilla: float
    msg_compress_vib_full: float  # %
    token_save_vib_full: float    # %
    mcnemar_p: float              # Full vs VIB paired (sampling baseline)
    full_wilson: Tuple[float, float]
    vib_wilson: Tuple[float, float]
    # Legacy t0-named batch fields; these do not certify effective request temperature.
    t0_delta_vib_full: Optional[float] = None
    t0_mcnemar_p: Optional[float] = None
    t0_msg_compress: Optional[float] = None
    # MBPP-only split blocks
    es_count: int = 0
    es_pct: float = 0.0
    es_vanilla: float = 0.0
    es_full: float = 0.0
    es_vib: float = 0.0
    es_msg_compress: float = 0.0
    mr_count: int = 0
    mr_pct: float = 0.0
    mr_vanilla: float = 0.0
    mr_full: float = 0.0
    mr_vib: float = 0.0
    mr_delta_vib_full: float = 0.0
    mr_msg_compress: float = 0.0


def load(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def validate_math_protocol(*runs: Dict[str, Any]) -> None:
    """Refuse to mix legacy labels with separately corrected MATH scores."""
    expected = None
    task_ids = None
    for run in runs:
        if run is None or "math" not in run.get("summary", {}):
            continue
        provenance = run.get("math_eval_correction", {}).get("provenance", {})
        fields = ("method", "matcher_sha256", "sympy_version")
        if not all(provenance.get(key) for key in fields):
            raise ValueError("MATH requires a provenance-checked whole-answer correction")
        protocol = tuple(provenance[key] for key in fields)
        ids = [row["task_id"] for row in run["detailed_results"]["math"]]
        if len(set(ids)) != len(ids) or len(ids) != run["summary"]["math"]["total_tasks"]:
            raise ValueError("MATH task IDs or counts are inconsistent")
        if expected is not None and (protocol != expected or set(ids) != task_ids):
            raise ValueError("MATH comparison protocols or task sets differ")
        expected, task_ids = protocol, set(ids)


def _build_row(ds: str, label: str, van: Dict[str, Any], main: Dict[str, Any],
               t0_json: Optional[Dict[str, Any]] = None) -> DatasetRow:
    """Assemble a DatasetRow by joining vanilla summary + main summary + analyze_multi CI."""
    v = van["summary"][ds]
    m = main["summary"][ds]
    detailed_rows = main["detailed_results"][ds]
    # Use analyze_dataset to get CI (same pipeline the Markdown report uses)
    simple_rows = [{
        "task_id": r["task_id"],
        "full_passed": r["full_passed"], "vib_passed": r["vib_passed"],
        "full_messages": r["full_messages"], "vib_messages": r["vib_messages"],
        "full_tokens": r["full_tokens"], "vib_tokens": r["vib_tokens"],
    } for r in detailed_rows]
    analysis = an.analyze_dataset(ds, simple_rows, m)

    delta_vf = analysis["pass_delta_pp"]
    delta_vf_ci = tuple(analysis["pass_delta_bootstrap_ci_95"])
    row = DatasetRow(
        ds=ds,
        label=label,
        total=m["total_tasks"],
        vanilla_pass=v["full_pass_rate"],
        full_pass=m["full_pass_rate"],
        vib_pass=m["vib_pass_rate"],
        delta_vib_full=delta_vf,
        delta_vib_full_ci=delta_vf_ci,
        delta_vib_vanilla=round(m["vib_pass_rate"] - v["full_pass_rate"], 2),
        msg_compress_vib_full=m["msg_compression_pct"],
        token_save_vib_full=m["token_saving_pct"],
        mcnemar_p=analysis["mcnemar_pvalue"],
        full_wilson=tuple(analysis["full_pass_wilson_ci_95"]),
        vib_wilson=tuple(analysis["vib_pass_wilson_ci_95"]),
    )
    row.t0_delta_vib_full = _t0_delta(t0_json, ds)
    row.t0_mcnemar_p = _t0_mcnemar_p(t0_json, ds)
    row.t0_msg_compress = _t0_msg_compress(t0_json, ds)
    # MBPP ES/MR splits: vanilla + main side by side
    if ds == "mbpp":
        row.es_count = m["early_stopped_count"]
        row.es_pct = m["early_stopped_pct"]
        row.es_vanilla = v.get("early_stopped_full_pass_rate", 0.0)
        row.es_full = m["early_stopped_full_pass_rate"]
        row.es_vib = m["early_stopped_vib_pass_rate"]
        row.es_msg_compress = m["early_stopped_msg_compression_pct"]
        row.mr_count = m["multi_round_count"]
        row.mr_pct = m["multi_round_pct"]
        row.mr_vanilla = v.get("multi_round_full_pass_rate", 0.0)
        row.mr_full = m["multi_round_full_pass_rate"]
        row.mr_vib = m["multi_round_vib_pass_rate"]
        row.mr_delta_vib_full = m["multi_round_pass_delta_pp"]
        row.mr_msg_compress = m["multi_round_msg_compression_pct"]
    return row


# ---------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------

def _fmt_ci(lo: float, hi: float) -> str:
    return rf"[{lo:+.2f},\,{hi:+.2f}]"


def _pp(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f}"


def _pct(x: float) -> str:
    return f"{x:.2f}"


def _pval(p: float) -> str:
    if p >= 1.0:
        return "1.00"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ---------------------------------------------------------------
# Legacy rerun-batch helpers
# ---------------------------------------------------------------

def _t0_mcnemar_p(t0_json: Optional[Dict[str, Any]], ds: str) -> Optional[float]:
    if t0_json is None:
        return None
    s = t0_json.get("summary", {}).get(ds)
    if s is None:
        return None
    b = s.get("vib_only_count", 0)
    c = s.get("full_only_count", 0)
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)


def _t0_delta(t0_json: Optional[Dict[str, Any]], ds: str) -> Optional[float]:
    if t0_json is None:
        return None
    s = t0_json.get("summary", {}).get(ds)
    if s is None:
        return None
    return s.get("pass_delta_pp")


def _t0_msg_compress(t0_json: Optional[Dict[str, Any]], ds: str) -> Optional[float]:
    if t0_json is None:
        return None
    s = t0_json.get("summary", {}).get(ds)
    if s is None:
        return None
    return s.get("msg_compression_pct")


def _star(_delta: float, ci: Tuple[float, float], mcnemar: float,
          _t0_mcnemar: Optional[float] = None, _t0_delta: Optional[float] = None) -> str:
    """Mark only primary-batch evidence; historical t0 labels do not verify temperature."""
    stars = ""
    lo, hi = ci
    if not (lo <= 0 <= hi):
        stars += r"$^\dagger$"
    if mcnemar < 0.05:
        stars += r"$^*$"
    return stars


# ---------------------------------------------------------------
# Table 1: Vanilla / Full / VIB × HumanEval · MBPP · GSM8K
# ---------------------------------------------------------------

TABLE1_HEADER = r"""\begin{table*}[t]
\centering
\caption{Observed accuracy and communication budgets: Vanilla single-agent, Full-Comm 3-agent, and VIB-Sparse.
Point estimates (pp) and 95\% paired-bootstrap CIs are reported for VIB vs Full; VIB vs Vanilla reports the raw pp difference without CI for brevity.
$\dagger$: 95\% CI does not span zero;
$*$: primary-batch McNemar $p<0.05$.
Tests are unadjusted. MBPP is excluded because of reference-assisted stopping (see \Cref{app:gate_trace}).
MATH reports conservative whole-answer matching, not a complete semantic oracle.
\label{tab:main_results}}
\setlength{\tabcolsep}{3.4pt}
\scriptsize
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{} l r r r r r r r r r @{}}
\toprule
\textbf{Dataset} & \textbf{N} &
\textbf{Vanilla} & \textbf{Full} & \textbf{VIB} &
\textbf{$\Delta_{\text{V-V}}$} &
\multicolumn{2}{c}{\textbf{$\Delta_{\text{V-F}}$}} &
\textbf{Msg} & \textbf{Tok} \\
& & & & & & \multicolumn{1}{c}{pp} & \multicolumn{1}{c}{95\% CI} & & \\
\cmidrule(lr){7-8}
& & \multicolumn{3}{c}{Pass@1 (\%)} & (pp) & & & \multicolumn{1}{c}{$\downarrow$\%} & \multicolumn{1}{c}{$\downarrow$\%} \\
\midrule
"""

TABLE1_FOOTER = r"""\bottomrule
\end{tabular}}
\vspace{0.3em}
\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{\sloppy\footnotesize
\textbf{Notes.} All intervals at 95\% level. Single-proportion intervals use Wilson score; method-difference intervals use 10k paired bootstrap draws (seed=20260812).
Vanilla uses direct synthesis without team discussion, with temperature$=0.1$ and task-specific output limits ($1024$ code, $256$ math). It is not matched to the team's total inference budget.
No equivalence or non-inferiority margin was prespecified. A CI spanning zero is not evidence of equivalence.
HumanEval and MBPP pass labels were recomputed with the harness code evaluator (see \Cref{app:repro}); an earlier artifact reported HumanEval at $100.00/100.00$ because the synthesis-refresh scoring step executed the reference tests without invoking them.
MATH labels use the same whole-answer matcher across all methods and recorded reruns; opaque forms can retain false negatives.
\textbf{Accounting.} Tok denotes historical counter reduction, not refreshed end-to-end cost.
HumanEval refreshed outputs retain old token fields, so its $3.88\%$ is not verified complete-run savings.
\textbf{Protocol audit.} Files labeled \texttt{t0} do not establish temperature zero: agent and synthesis calls override client defaults. Reruns are descriptive, not temperature-robustness evidence.
__MATH_NOTE__}
\end{table*}
"""


def _main_row_lines(rows: List[DatasetRow]):
    """Share every numeric row between the English and Chinese manuscripts."""
    for r in rows:
        if r.ds == "mbpp":
            continue
        star = _star(r.delta_vib_full, r.delta_vib_full_ci, r.mcnemar_p, r.t0_mcnemar_p,
                     r.t0_delta_vib_full)
        yield (
            f"{r.label:<15} & {r.total:>5} & "
            f"{_pct(r.vanilla_pass):>7} & "
            f"{_pct(r.full_pass)}~{_fmt_ci(*r.full_wilson):>14} & "
            f"{_pct(r.vib_pass)}~{_fmt_ci(*r.vib_wilson):>14} & "
            f"{_pp(r.delta_vib_vanilla):>6} & "
            f"{_pp(r.delta_vib_full)}{star:>6} & {_fmt_ci(*r.delta_vib_full_ci):>14} & "
            f"{_pp(r.msg_compress_vib_full):>6} & "
            f"{_pp(r.token_save_vib_full):>6} \\\\"
        )


def render_table1(rows: List[DatasetRow]) -> str:
    lines = [TABLE1_HEADER, *_main_row_lines(rows)]
    math_row = next((row for row in rows if row.ds == "math"), None)
    math_note = ""
    if math_row is not None:
        math_note = (
            rf"\textbf{{MATH matching results.}} Primary-batch difference "
            rf"${math_row.delta_vib_full:+.2f}$pp (McNemar $p={_pval(math_row.mcnemar_p)}$)"
        )
        if math_row.t0_delta_vib_full is not None:
            math_note += (
                rf", recorded rerun ${math_row.t0_delta_vib_full:+.2f}$pp "
                rf"($p={_pval(math_row.t0_mcnemar_p)}$)"
            )
        math_note += "; these are conditional on the matching protocol, not universal accuracy guarantees."
    lines.append(TABLE1_FOOTER.replace("__MATH_NOTE__", math_note))
    return "\n".join(lines)


def render_table1_chinese(rows: List[DatasetRow]) -> str:
    """Render a localized main table without translating or copying stored numeric data."""
    header = r"""\begin{table*}[t]
\centering
\caption{三种方法的历史表现与通信预算。VIB相对Full的差值以百分点及配对bootstrap 95\%区间表示；相对Vanilla仅报告点估计。$\dagger$表示区间不跨零；$*$表示主批次McNemar检验$p<0.05$，未经多重检验校正。MBPP因参考测试辅助早停而排除。MATH为保守完整答案匹配率。\label{tab:main_results}}
"""
    layout = TABLE1_HEADER[TABLE1_HEADER.index(r"\setlength"):]
    for english, chinese in (
        (r"\textbf{Dataset}", r"\textbf{数据集}"),
        (r"\textbf{N}", r"\textbf{题数}"),
        (r"\textbf{Msg}", r"\textbf{广播}"),
        (r"\textbf{Tok}", r"\textbf{历史Token}"),
    ):
        layout = layout.replace(english, chinese)
    footer = r"""\bottomrule
\end{tabular}}
\vspace{0.3em}
\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{\sloppy\footnotesize
\textbf{口径说明。}单比例区间使用Wilson方法，差值区间使用10000次任务配对bootstrap。Vanilla的推理预算不与团队对齐；区间跨零不证明等效或非劣效。历史计数减少不等于最终运行的完整费用：HumanEval刷新代码继承旧Token字段，其3.88\%不能作为刷新后完整节省。文件名中的\texttt{t0}不验证真实温度零，不能据此声称温度稳健性。MATH未知格式仍可能漏判。}
\end{table*}
"""
    return "\n".join([header + layout, *_main_row_lines(rows), footer])


# ---------------------------------------------------------------
# Table 2: MBPP Early-Stopped vs Multi-Round
# ---------------------------------------------------------------

TABLE2_HEADER = r"""\begin{table}[t]
\centering
\caption{Historical MBPP oracle-only results, excluded from headline comparisons.
ES = stopped by reference-test evaluation after one-round team synthesis; MR = not stopped.
The OR partition depends on recorded outcomes; discarded probe costs may be missing.
\label{tab:mbpp_split}}
\setlength{\tabcolsep}{2.8pt}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{} >{\raggedright\arraybackslash}p{2.2cm} r rrr r r @{}}
\toprule
\textbf{Group} & \textbf{\makecell{N\\(\%)}} &
\textbf{Van.} & \textbf{Full} & \textbf{VIB} &
\textbf{$\Delta_{\text{V-F}}$} &
\textbf{Msg$\downarrow$} \\
& & \multicolumn{3}{c}{Pass@1 (\%)} & (pp) & (\%) \\
\midrule
"""

TABLE2_FOOTER = r"""\bottomrule
\end{tabular}}
\end{table}
"""


def render_table2(mbpp: DatasetRow) -> str:
    lines = [TABLE2_HEADER]
    lines.append(
        f"All (oracle-only) & {mbpp.total} (100.00\\%) & "
        f"{_pct(mbpp.vanilla_pass)} & {_pct(mbpp.full_pass)} & {_pct(mbpp.vib_pass)} & "
        f"{_pp(mbpp.delta_vib_full)} & {_pp(mbpp.msg_compress_vib_full)} \\\\"
    )
    lines.append(
        f"Early-Stopped & {mbpp.es_count} ({_pct(mbpp.es_pct)}\\%) & "
        f"{_pct(mbpp.es_vanilla):>6} & "
        f"{_pct(mbpp.es_full):>6} & "
        f"{_pct(mbpp.es_vib):>6} & "
        f"{_pp(round(mbpp.es_vib - mbpp.es_full, 2)):>6} & "
        f"{_pp(mbpp.es_msg_compress):>6} \\\\"
    )
    lines.append(
        f"Multi-Round    & {mbpp.mr_count} ({_pct(mbpp.mr_pct)}\\%) & "
        f"{_pct(mbpp.mr_vanilla):>6} & "
        f"{_pct(mbpp.mr_full):>6} & "
        f"{_pct(mbpp.mr_vib):>6} & "
        f"{_pp(mbpp.mr_delta_vib_full):>6} & "
        f"{_pp(mbpp.mr_msg_compress):>6} \\\\"
    )
    lines.append(TABLE2_FOOTER)
    return "\n".join(lines)


# ---------------------------------------------------------------
# Table 3: GSM8K Ablation: frozen-only vs frozen+role-weights
# ---------------------------------------------------------------

TABLE3_HEADER = r"""\begin{table}[t]
\centering
\caption{GSM8K ablation on $N=200$ sub-sample toggling role-aware weights (Section 3, Method).
Message reduction is $30.25\%$ without role weights and $25.17\%$ with them.
These are separate stochastic executions; the Full baseline also varies by $1$~pp.
No formal non-inferiority or role-specific accuracy benefit is established.
\label{tab:gsm8k_ablation}}
\setlength{\tabcolsep}{3.2pt}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{} >{\raggedright\arraybackslash}p{3.1cm} r r r r r r @{}}
\toprule
\textbf{Ablation} & \textbf{N} &
\textbf{Full} & \textbf{VIB} & \textbf{$\Delta_{\text{V-F}}$} &
\textbf{Msg$\downarrow$} & \textbf{Tok$\downarrow$} \\
\midrule
"""

TABLE3_FOOTER = r"""\bottomrule
\end{tabular}}
\end{table}
"""


def render_table3(ablation_only: Dict[str, Any], ablation_role: Dict[str, Any]) -> str:
    def _row(name: str, abl: Dict[str, Any]) -> str:
        s = abl["summary"]["gsm8k"]
        return (
            f"{name:<26} & {s['total_tasks']:>4} & "
            f"{_pct(s['full_pass_rate']):>6} & "
            f"{_pct(s['vib_pass_rate']):>6} & "
            f"{_pp(s['pass_delta_pp']):>6} & "
            f"{_pp(s['msg_compression_pct']):>6} & "
            f"{_pp(s['token_saving_pct']):>6} \\\\"
        )

    lines = [TABLE3_HEADER]
    lines.append(_row("GSM8K $\\cdot$ frozen\\_only", ablation_only))
    lines.append(_row("GSM8K $\\cdot$ frozen+role", ablation_role))
    lines.append(TABLE3_FOOTER)
    return "\n".join(lines)


# ---------------------------------------------------------------
# Table 5: Cross-family transfer (matched harness config, N=200)
# ---------------------------------------------------------------

TABLE5_CAPTION = r"""\caption{Cross-family transfer on GSM8K ($N=200$).
Both rows run under the \emph{identical} harness configuration (paper defaults $\beta_0=3.2$, $\lambda=0.08$, $\gamma=-0.4$,
agent-turn temperature $0.2$ and synthesis temperature $0.1$), using different gateway model IDs.
Message reduction exceeds $25\%$ and accuracy point estimates are positive in both runs.
This is limited transfer evidence, not a formal non-inferiority result or verification of immutable served snapshots.
\label{tab:crossmodel}}"""

TABLE5_NOTE = r"""\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{\sloppy\footnotesize
\textbf{Reading the token column.} VIB compresses the \emph{shared context}, not the model's output:
prompt tokens fall by $4.6$--$5.2\%$, while completion tokens rise
($+0.45\%$ DeepSeek, $+3.64\%$ Qwen) and the
\emph{net} token saving ($1.2$--$1.7\%$) trails the message/prompt reduction.
These aggregate observations are consistent with output-length rebound but do not by themselves
establish that shorter context caused longer reasoning. Token counts are not a measured monetary or latency saving.}"""


def render_table5_crossmodel(deepseek: Dict[str, Any], qwen: Dict[str, Any]) -> str:
    def _row(name: str, run: Dict[str, Any]) -> str:
        s = run["summary"]["gsm8k"]
        return (
            f"{name} & {_pct(s['full_pass_rate'])} & {_pct(s['vib_pass_rate'])} & "
            f"{_pp(s['pass_delta_pp'])} & {_pp(s['msg_compression_pct'])} & "
            f"{_pp(s['prompt_saving_pct'])} & {_pp(s['token_saving_pct'])} \\\\"
        )

    lines = [
        r"""\begin{table}[t]
\centering
""" + TABLE5_CAPTION + r"""
\setlength{\tabcolsep}{3.2pt}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{} >{\raggedright\arraybackslash}p{3.3cm} r r r r r r @{}}
\toprule
\textbf{Backbone} & \textbf{Full} & \textbf{VIB} & \textbf{$\Delta_{\text{V-F}}$} &
\textbf{Msg$\downarrow$} & \textbf{Prompt$\downarrow$} & \textbf{Tok$\downarrow$} \\
\midrule
""",
        _row("DeepSeek v4-pro", deepseek),
        _row("Qwen-3.8-Max", qwen),
        r"""\bottomrule
\end{tabular}}
""",
        TABLE5_NOTE,
        r"""\end{table}
""",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------
# Table 4: OFAT hyperparameter sensitivity (GSM8K N=200)
# ---------------------------------------------------------------

# The sweep grid is OFAT with the default triple repeated three times (once per
# parameter axis). Those three repeats are the *same* configuration, so their
# spread is a free run-to-run noise estimate at N=200 — we surface it instead of
# hiding it behind a single anchor row.
SENSITIVITY_GRID_SIZE = 15
# Only accuracy is graded as "stable": the whole point of the appendix is that the
# headline conclusion (VIB ≈ Full) survives any reasonable hyperparameter choice.
# Message reduction is *expected* to move with beta_0 (that scalar trades
# sufficiency against redundancy), so folding it into the flag would blank every
# row on the beta_0 axis and hide the actual finding.
STABLE_DELTA_PP = 1.0


def _sensitivity_ready(sweep: Dict[str, Any]) -> str:
    """Return "" when the sweep is complete, else a human-readable reason to skip."""
    if sweep.get("aborted"):
        return "sweep_summary.json 标记 aborted，未跑完"
    rows = sweep.get("rows") or []
    if len(rows) != SENSITIVITY_GRID_SIZE:
        return f"仅 {len(rows)}/{SENSITIVITY_GRID_SIZE} 个网格点"
    empty = [r["name"] for r in rows if not (r.get("summary") or {}).get("total_tasks")]
    if empty:
        return f"{len(empty)} 个点缺少 summary: {', '.join(empty[:3])}"
    return ""


def render_table4_sensitivity(sweep: Dict[str, Any]) -> str:
    anchor, rows = sweep["anchor"], sweep["rows"]
    reps = [r["summary"] for r in rows
            if r["param"] in anchor and float(r["value"]) == float(anchor[r["param"]])]

    def _mean(key: str) -> float:
        return sum(float(s[key]) for s in reps) / len(reps)

    a_delta = _mean("pass_delta_pp")
    rep_spread_delta = max(float(s["pass_delta_pp"]) for s in reps) - \
                       min(float(s["pass_delta_pp"]) for s in reps)
    rep_spread_msg = max(float(s["msg_compression_pct"]) for s in reps) - \
                     min(float(s["msg_compression_pct"]) for s in reps)

    def _dev(key: str, ref: float) -> float:
        return max(abs(float(r["summary"][key]) - ref) for r in rows)

    max_dev_delta = _dev("pass_delta_pp", a_delta)

    # Message reduction is not just a beta_0 story: gamma (the admission threshold)
    # is an equally strong knob in the opposite direction, while lambda barely
    # moves it. Report each axis' own span instead of lumping them together, and
    # only claim monotonicity when the observed sequence actually is monotone.
    def _axis_msg(param: str) -> List[float]:
        return [float(r["summary"]["msg_compression_pct"]) for r in rows if r["param"] == param]

    def _monotone(vals: List[float]) -> str:
        if all(b <= a for a, b in zip(vals, vals[1:])):
            return "decreasing"
        if all(b >= a for a, b in zip(vals, vals[1:])):
            return "increasing"
        return "non-monotone"

    beta_msg, lambda_msg, gamma_msg = _axis_msg("beta0"), _axis_msg("lambda"), _axis_msg("gamma")

    labels = {"beta0": r"$\beta_0$", "lambda": r"$\lambda$", "gamma": r"$\gamma$"}
    order = [("beta0", labels["beta0"]), ("lambda", labels["lambda"]), ("gamma", labels["gamma"])]

    body = []
    for param, label in order:
        for r in [x for x in rows if x["param"] == param]:
            s, val = r["summary"], r["value"]
            is_default = float(val) == float(anchor[param])
            dpp = float(s["pass_delta_pp"])
            msg = float(s["msg_compression_pct"])
            stable = abs(dpp - a_delta) <= STABLE_DELTA_PP
            fmt_d = (r"\textbf{%s}" % _pp(dpp)) if is_default else _pp(dpp)
            fmt_m = (r"\textbf{%s}" % _pct(msg)) if is_default else _pct(msg)
            fmt_t = (r"\textbf{%s}" % _pct(s["token_saving_pct"])) if is_default \
                else _pct(s["token_saving_pct"])
            fmt_v = (r"\textbf{%s}" % _pct(val)) if is_default else _pct(val)
            body.append(f"{label} & {fmt_v} & {fmt_d} & {fmt_m} & {fmt_t} & "
                        f"{r'\checkmark' if stable else ''} \\\\")

    model = sweep.get("model") or "unknown"
    # Build the header separately: it is LaTeX (full of `\%` and `{!}`) that would
    # collide with %-formatting, so the model id is substituted textually.
    header = r"""\begin{table}[t]
\centering
\caption{One-factor-at-a-time sensitivity to the three interpretable VIB scalars on GSM8K ($N=200$, \texttt{__MODEL__}).
Each block varies one parameter while the other two stay at the paper defaults
$\beta_0=3.2$, $\lambda=0.08$, $\gamma=-0.4$ (bolded rows); these three bolded rows are the \emph{same}
configuration, so their spread is a descriptive run-to-run observation at this $N$, not a noise bound.
Columns: VIB$-$Full Pass@1 $\Delta$pp, inter-agent message reduction, total-token reduction;
$\Delta$~ok marks rows whose accuracy gap stays within $\pm1$ pp of the mean of the three default repeats.
\label{tab:sensitivity}}
\setlength{\tabcolsep}{3.2pt}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{} l r r r r c @{}}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{$\Delta$pp} & \textbf{Msg$\downarrow$\%} &
\textbf{Tok$\downarrow$\%} & \textbf{$\Delta$~ok} \\
\midrule
""".replace("__MODEL__", model)

    lines = [
        header,
        "\n".join(body),
        r"""\bottomrule
\end{tabular}}
\vspace{0.3em}
\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{\sloppy\footnotesize
\textbf{Interpretation.} Across these %d grid points, the observed
VIB$-$Full $\Delta$ deviates from the mean default by at most $%.2f$ pp.
This fixed-subset, one-factor comparison does not establish non-inferiority or joint-parameter robustness.
Observed message reduction varies across the gate settings: $\beta_0$ spans
$%.1f$\%%--$%.1f$\%% (%s as it grows) and $\gamma$ spans $%.1f$\%%--$%.1f$\%% (%s), whereas
$\lambda$ shifts it over $%.1f$\%%--$%.1f$\%% across a threefold range.
We retain the main defaults $\beta_0=3.2$, $\lambda=0.08$, $\gamma=-0.4$;
deployment calibration requires independent tasks and complete cost accounting.
The three bolded rows repeat the \emph{same} configuration and show ranges of
$%.2f$ pp in $\Delta$ and $%.2f$ pp in message reduction.
These three executions are not a confidence bound on future runs.}""" % (
            len(rows), max_dev_delta,
            min(beta_msg), max(beta_msg), _monotone(beta_msg),
            min(gamma_msg), max(gamma_msg), _monotone(gamma_msg),
            min(lambda_msg), max(lambda_msg),
            rep_spread_delta, rep_spread_msg),
        r"""\end{table}
""",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------
# Table 6: naive-gating controls (Appendix D)
# ---------------------------------------------------------------

GATING_POINT_COUNT = 6  # 1 VIB anchor + 1 random-drop + 4 diversity top-k


def _gating_ready(sweep: Dict[str, Any]) -> str:
    """Return "" when the gating sweep is complete, else a reason to skip."""
    if sweep.get("aborted"):
        return "gating_summary.json 标记 aborted，未跑完"
    rows = sweep.get("rows") or []
    if len(rows) != GATING_POINT_COUNT:
        return f"仅 {len(rows)}/{GATING_POINT_COUNT} 个点"
    empty = [r["name"] for r in rows if not (r.get("summary") or {}).get("total_tasks")]
    if empty:
        return f"{len(empty)} 个点缺少 summary: {', '.join(empty[:3])}"
    return ""


def _detailed_rows(point: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (point.get("detailed_results") or {}).get("gsm8k") or []


def _gating_outcomes(point: Dict[str, Any], outcome_key: str) -> Dict[str, bool]:
    """拒绝重复或缺失任务，避免配对统计被静默缩减。"""
    outcomes = {}
    for row in _detailed_rows(point):
        task_id = row.get("task_id")
        if not task_id or task_id in outcomes:
            raise ValueError(f"Missing or duplicate gating task id: {task_id!r}")
        outcome = row.get(outcome_key)
        if not isinstance(outcome, bool):
            raise ValueError(f"Invalid {outcome_key} for task {task_id}")
        outcomes[task_id] = outcome
    if not outcomes:
        raise ValueError("No gating task outcomes")
    return outcomes


def _paired_vs_anchor(anchor: Dict[str, bool], arm: Dict[str, bool],
                      n_boot: int = 10000, seed: int = 20260812):
    """Paired bootstrap CI for (arm − anchor) in pp over the shared task ids.

    Reuses the manuscript's single bootstrap implementation: the helper consumes
    ``full_passed``/``vib_passed`` keys, so the anchor is mapped onto the former
    and the comparison arm onto the latter.
    """
    if not anchor or anchor.keys() != arm.keys():
        raise ValueError("Gating arms must contain identical nonempty task sets")
    rows = [{"full_passed": anchor[task_id], "vib_passed": arm[task_id]}
            for task_id in anchor]
    return an._paired_delta_pp_bootstrap_ci(rows, n_boot=n_boot, seed=seed)


def render_table6_gating(sweep: Dict[str, Any], gating_dir: Path) -> str:
    rows = sweep["rows"]
    by_arm = {r["arm"]: r for r in rows}
    anchor = by_arm["vib"]

    anchor_json = load(gating_dir / f"{anchor['name']}.json")
    anchor_vib = _gating_outcomes(anchor_json, "vib_passed")
    anchor_full = _gating_outcomes(anchor_json, "full_passed")
    n = len(anchor_vib)

    keep_rate = float(sweep.get("matched_keep_rate") or 0.0)
    model = sweep.get("model") or "unknown"

    # (label, msg-down%, tok-down%, pass%, delta_pp, ci)
    body_rows = []

    full_pct = float(anchor["summary"]["full_pass_rate"])
    full_delta = float(anchor["summary"]["pass_delta_pp"]) * -1.0
    full_ci = _paired_vs_anchor(anchor_vib, anchor_full)
    body_rows.append(("No gate (Full)", None, None, full_pct, full_delta, full_ci, False))

    vib_pct = float(anchor["summary"]["vib_pass_rate"])
    body_rows.append(("VIB (anchor)", float(anchor["summary"]["msg_compression_pct"]),
                      float(anchor["summary"]["token_saving_pct"]), vib_pct, 0.0, (0.0, 0.0), True))

    naive_labels = {
        "random": f"Random drop ($p={keep_rate:.3f}$)",
    }
    for r in rows:
        if r["arm"] == "vib":
            continue
        point = load(gating_dir / f"{r['name']}.json")
        arm_map = _gating_outcomes(point, "vib_passed")
        paired_ci = _paired_vs_anchor(anchor_vib, arm_map)
        label = naive_labels.get(r["arm"]) or f"Cosine threshold ($\\theta={float(r['knob']):.2f}$)"
        delta = 100.0 * sum(arm_map[task_id] - anchor_vib[task_id]
                           for task_id in anchor_vib) / n
        body_rows.append((
            label,
            float(r["summary"]["msg_compression_pct"]),
            float(r["summary"]["token_saving_pct"]),
            float(r["summary"]["vib_pass_rate"]),
            delta,
            paired_ci,
            False,
        ))

    def _pct_cell(x):
        return "--" if x is None else _pct(x)

    def _delta_cell(delta, ci):
        return f"${delta:+.2f}$~$[{ci[0]:+.2f}, {ci[1]:+.2f}]$"

    body = []
    for label, msg, tok, pct, delta, ci, is_anchor in body_rows:
        if is_anchor:
            cells = (r"\textbf{%s}" % label, r"\textbf{%s}" % _pct(msg),
                     r"\textbf{%s}" % _pct(tok), r"\textbf{%s}" % _pct(pct),
                     r"\textbf{%s}" % _delta_cell(delta, ci))
        else:
            cells = (label, _pct_cell(msg), _pct_cell(tok), _pct(pct), _delta_cell(delta, ci))
        body.append(" & ".join(cells) + r" \\")

    # Report the realised budget match rather than asserting it by hand: the naive
    # arms are calibrated offline, so the compression they actually pay can drift
    # from the anchor's by a little. Only the arms deliberately pinned to the
    # anchor's budget count here -- the random drop and the nearest top-k point.
    # The wider top-k settings remove less on purpose and trace the trend instead.
    vib_comp = float(anchor["summary"]["msg_compression_pct"])
    naive = [r for r in rows
             if r["arm"] != "vib" and (r.get("summary") or {}).get("total_tasks")]
    topk = [r for r in naive if r["arm"] == "topk"]
    matched = [r for r in naive if r["arm"] != "topk"]
    if topk:
        matched.append(min(topk, key=lambda r: abs(
            float(r["summary"]["msg_compression_pct"]) - vib_comp)))
    budget_span = max((abs(float(r["summary"]["msg_compression_pct"]) - vib_comp)
                       for r in matched), default=0.0)

    header = r"""\begin{table}[t]
\centering
\caption{Lightweight gating controls using the same gateway model ID
(\texttt{__MODEL__}), $200$ GSM8K tasks, and harness, but separate stochastic executions.
\emph{Random drop} uses probability $p$ calibrated to VIB admission, and the \emph{cosine-threshold gate} drops any candidate whose
maximum cosine against the already-broadcast history exceeds $\theta$ --- it reuses the same frozen
encoder but omits the goal term, schedule, and VIB-specific corrections. This is threshold admission, not fixed-cardinality top-$k$.
$\Delta$ is the paired (arm$-$VIB) Pass@1 difference in pp with a $10\,000$-resample bootstrap 95\% CI;
negative means the arm is worse than VIB. The arms pinned to the anchor's message budget --- the random
drop and the nearest cosine threshold --- realise a reduction within $__BUDGET__$~pp of the anchor's; other
thresholds trace different budgets. Matching messages does not match total-token cost.
Intervals spanning zero do not establish equivalence or VIB-specific superiority.
\label{tab:gating_controls}}
\setlength{\tabcolsep}{3.0pt}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{} l r r r r @{}}
\toprule
\textbf{Gate} & \textbf{Msg$\downarrow$\%} & \textbf{Tok$\downarrow$\%} & \textbf{Pass@1} &
\textbf{$\Delta$ vs VIB (pp)} \\
\midrule
""".replace("__MODEL__", model).replace("__BUDGET__", f"{budget_span:.1f}")

    footer = r"""\bottomrule
\end{tabular}}
\end{table}
"""
    return header + "\n".join(body) + "\n" + footer


# ---------------------------------------------------------------
# README
# ---------------------------------------------------------------

README = r"""
此目录下表格为 scripts/export_latex_tables.py 自动生成，请勿手动编辑。

在论文中引用：
    \input{paper/tables/table1_main_three_methods}
    \input{paper/tables/table2_mbpp_es_mr_split}
    \input{paper/tables/table3_gsm8k_ablation}
    \input{paper/tables/table4_sensitivity}   % Appendix B: hyperparameter sensitivity
    \input{paper/tables/table5_crossmodel}    % Appendix C: cross-family transfer
    \input{paper/tables/table6_gating_controls}  % Appendix D: naive-gating controls

若需重生成：
    python3 scripts/export_latex_tables.py

table4 需要 results/sweep_v1/sweep_summary.json 覆盖全部 15 个网格点，否则自动跳过。
table6 需要 results/gating_sweep/ 下 6 个点 (anchor + random + 4×top-k) 全部跑完，否则自动跳过。
"""


def preview_table1_plain(rows: List[DatasetRow]):
    hdr = (f"| {'Dataset':<15} | {'N':>5} | {'Vanilla':>9} | {'Full':>18} | {'VIB (Ours)':>18} | "
           f"{'ΔV-Van':>7} | {'ΔV-Full':>10} | {'95%CI':>18} | {'Msg↓%':>7} | {'Tok↓%':>7} |")
    print("\n" + "=" * len(hdr))
    print("Table 1 · 主结果预览（plain-text，LaTeX 见 paper/tables/）")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        star = _star(r.delta_vib_full, r.delta_vib_full_ci, r.mcnemar_p, r.t0_mcnemar_p,
                     r.t0_delta_vib_full)
        print(f"| {r.label:<15} | {r.total:>5} | {r.vanilla_pass:>7.2f}% | "
              f"{r.full_pass:>7.2f}% {_fmt_ci(*r.full_wilson):>10} | "
              f"{r.vib_pass:>7.2f}% {_fmt_ci(*r.vib_wilson):>10} | "
              f"{r.delta_vib_vanilla:>+6.2f} | "
              f"{r.delta_vib_full:>+6.2f}{star:>4} | "
              f"{_fmt_ci(*r.delta_vib_full_ci):>18} | "
              f"{r.msg_compress_vib_full:>+6.2f} | {r.token_save_vib_full:>+6.2f} |")
    print("=" * len(hdr))
    # Compare recorded batches without inferring request-level temperature.
    any_t0 = any(r.t0_mcnemar_p is not None for r in rows)
    if any_t0:
        print("\n── 历史标注 t0 的批次对照（不验证请求温度）──")
        for r in rows:
            if r.t0_mcnemar_p is None:
                continue
            delta_rng = (min(r.delta_vib_full, r.t0_delta_vib_full),
                         max(r.delta_vib_full, r.t0_delta_vib_full))
            p_rng = (min(r.mcnemar_p, r.t0_mcnemar_p),
                     max(r.mcnemar_p, r.t0_mcnemar_p))
            comp_delta = (r.t0_msg_compress - r.msg_compress_vib_full
                          if r.t0_msg_compress is not None else None)
            sig = _star(r.delta_vib_full, r.delta_vib_full_ci, r.mcnemar_p, r.t0_mcnemar_p,
                        r.t0_delta_vib_full)
            print(f"  · {r.label:<12} Δ(VIB−Full) 主批次={r.delta_vib_full:+.2f}pp / t0文件={r.t0_delta_vib_full:+.2f}pp  "
                  f"range=[{delta_rng[0]:+.2f},{delta_rng[1]:+.2f}]  "
                  f"McNemar p range=[{p_rng[0]:.4f},{p_rng[1]:.4f}]  "
                  f"marker={sig if sig else '(无)'}  "
                  f"compress Δ(t0−main)={'—' if comp_delta is None else f'{comp_delta:+.2f}pp'}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Export VIB-Sparse paper LaTeX tables from result JSONs")
    p.add_argument("--vanilla", default="results/vanilla_with_math_merged.matheval_fixed.json")
    p.add_argument("--main", default="results/main_with_math_merged.codeeval_fixed.matheval_fixed.json",
                   help="主结果 JSON；默认使用 scripts/recompute_code_eval.py 修正过"
                        "代码基准通过标记的产物（原始文件保持只读）")
    p.add_argument("--ablfrozen", default="results/gsm8k_ablation_frozen_only.json")
    p.add_argument("--ablrole", default="results/gsm8k_ablation_frozen_plus_role.json")
    p.add_argument("--outdir", default="paper/tables")
    p.add_argument("--t0-math", default="results/math_main_5000_t0_v2.matheval_fixed.json",
                   help="历史标注 t0 的 MATH 重跑，仅作批次描述，不验证请求温度")
    p.add_argument("--t0-gsm8k", default="results/gsm8k_main_1319_t0.json",
                   help="历史标注 t0 的 GSM8K 重跑，仅作批次描述，不验证请求温度")
    p.add_argument("--crossmodel", default="results/paper_improvement_crossmodel_qwen3_gsm8k_n200.json",
                   help="Table 5: 跨族模型 GSM8K N=200 结果 (Qwen-3.8-Max)")
    p.add_argument("--crossmodel-ref", default="results/sweep_gateway_v4pro/sweep_beta0_3p2.json",
                   help="Table 5 的 DeepSeek 行: 必须与 --crossmodel 使用同一 harness 配置。"
                        "取私有网关 deepseek-v4-pro-0817 的锚点，与 Qwen-3.8-Max 同为 gateway 部署，"
                        "从而把 backbone 族作为唯一变量")
    p.add_argument("--sweep", default="results/sweep_v1/sweep_summary.json",
                   help="Table 4: OFAT 灵敏度 sweep 汇总 (run_hyper_sweep.py 产出)，"
                        "底座须与正文主表一致 (deepseek-chat)")
    p.add_argument("--gating", default="results/gating_sweep/gating_summary.json",
                   help="Table 6: 朴素门控对照 sweep 汇总 (run_gating_baseline_sweep.py 产出)；"
                        "各点原始 JSON 需与其同目录，用于配对 bootstrap")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    van = load(Path(args.vanilla))
    main = load(Path(args.main))
    abl1 = load(Path(args.ablfrozen))
    abl2 = load(Path(args.ablrole))

    t0_math = None
    if args.t0_math and Path(args.t0_math).exists():
        t0_math = load(Path(args.t0_math))
    t0_gsm8k = None
    if args.t0_gsm8k and Path(args.t0_gsm8k).exists():
        t0_gsm8k = load(Path(args.t0_gsm8k))

    def _t0_for(ds: str) -> Optional[Dict[str, Any]]:
        if ds == "math":
            return t0_math
        if ds == "gsm8k":
            return t0_gsm8k
        return None

    validate_math_protocol(van, main, t0_math)
    rows = [
        _build_row("humaneval", "HumanEval", van, main, _t0_for("humaneval")),
        _build_row("mbpp", "MBPP", van, main, _t0_for("mbpp")),
        _build_row("gsm8k", "GSM8K", van, main, _t0_for("gsm8k")),
    ]
    if "math" in main.get("summary", {}):
        rows.append(_build_row("math", "MATH", van, main, _t0_for("math")))

    headline_rows = [row for row in rows if row.ds != "mbpp"]
    mbpp = next(row for row in rows if row.ds == "mbpp")
    t1 = render_table1(headline_rows)
    (outdir / "table1_main_three_methods.tex").write_text(t1)
    (outdir / "table1_main_three_methods_zh.tex").write_text(
        render_table1_chinese(headline_rows))
    (outdir / "table2_mbpp_es_mr_split.tex").write_text(render_table2(mbpp))
    (outdir / "table3_gsm8k_ablation.tex").write_text(render_table3(abl1, abl2))
    (outdir / "_README.txt").write_text(README)
    written = ["table1_main_three_methods.tex", "table1_main_three_methods_zh.tex",
               "table2_mbpp_es_mr_split.tex",
               "table3_gsm8k_ablation.tex"]

    # Table 5 is only meaningful as a *matched-config* contrast: the two backbones
    # must have run under an identical harness configuration, otherwise the delta
    # measures the config change rather than the model family.
    xm_main_p, xm_ref_p = Path(args.crossmodel), Path(args.crossmodel_ref)
    if xm_main_p.exists() and xm_ref_p.exists():
        xm_main, xm_ref = load(xm_main_p), load(xm_ref_p)
        cfg_main, cfg_ref = xm_main.get("experiment_config") or {}, xm_ref.get("experiment_config") or {}
        if cfg_main != cfg_ref:
            print("⚠️  Table 5 跳过: 两行 experiment_config 不一致，非 matched-config 对比")
            print(f"    {xm_main_p.name}: {cfg_main}")
            print(f"    {xm_ref_p.name}: {cfg_ref}")
        else:
            (outdir / "table5_crossmodel.tex").write_text(render_table5_crossmodel(xm_ref, xm_main))
            written.append("table5_crossmodel.tex")
    else:
        missing = ", ".join(str(p) for p in (xm_main_p, xm_ref_p) if not p.exists())
        print(f"⚠️  Table 5 跳过: 缺少 {missing}")

    # Table 4 needs the *whole* grid: a partially finished sweep would render an
    # appendix table that silently misrepresents the sensitivity landscape.
    sweep_p = Path(args.sweep)
    if not sweep_p.exists():
        print(f"⚠️  Table 4 跳过: 缺少 {sweep_p}")
    else:
        sweep = load(sweep_p)
        reason = _sensitivity_ready(sweep)
        if reason:
            print(f"⚠️  Table 4 跳过: {reason}")
        else:
            (outdir / "table4_sensitivity.tex").write_text(render_table4_sensitivity(sweep))
            written.append("table4_sensitivity.tex")

    # Table 6 is a control experiment: a half-finished sweep would understate the
    # naive arms, so require every point to be present and un-aborted.
    gating_p = Path(args.gating)
    if not gating_p.exists():
        print(f"⚠️  Table 6 跳过: 缺少 {gating_p}")
    else:
        gating = load(gating_p)
        reason = _gating_ready(gating)
        if reason:
            print(f"⚠️  Table 6 跳过: {reason}")
        else:
            (outdir / "table6_gating_controls.tex").write_text(
                render_table6_gating(gating, gating_p.parent))
            written.append("table6_gating_controls.tex")

    preview_table1_plain(headline_rows)
    print(f"\n✅ 已生成 {len(written)} 张 .tex + 1 README to {outdir.resolve()}:")
    for name in written:
        f = outdir / name
        print("   ·", f.name, f"({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

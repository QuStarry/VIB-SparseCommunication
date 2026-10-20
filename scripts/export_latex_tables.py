r"""
Phase 3-a: 把 3 类实验 JSON 结果自动导出为 LaTeX 顶会论文表格。

Inputs (默认路径，CLI 可覆盖):
  --vanilla     results/vanilla_with_math_merged.json    (含 MATH，Table 1 第 4 行必需)
  --main        results/main_with_math_merged.json       (含 MATH，Table 1 第 4 行必需)
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
    # t0 (temperature=0) robustness fields — filled only if --t0-* JSON is provided
    t0_delta_vib_full: Optional[float] = None
    t0_mcnemar_p: Optional[float] = None
    t0_msg_compress: Optional[float] = None
    # MBPP-only split blocks
    es_count: int = 0
    es_pct: float = 0.0
    es_vanilla: float = 0.0
    es_full: float = 0.0
    es_vib: float = 0.0
    mr_count: int = 0
    mr_pct: float = 0.0
    mr_vanilla: float = 0.0
    mr_full: float = 0.0
    mr_vib: float = 0.0
    mr_delta_vib_full: float = 0.0


def load(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


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
        row.mr_count = m["multi_round_count"]
        row.mr_pct = m["multi_round_pct"]
        row.mr_vanilla = v.get("multi_round_full_pass_rate", 0.0)
        row.mr_full = m["multi_round_full_pass_rate"]
        row.mr_vib = m["multi_round_vib_pass_rate"]
        row.mr_delta_vib_full = m["multi_round_pass_delta_pp"]
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
# t0 (temperature=0) robustness helpers
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


def _star(delta: float, ci: Tuple[float, float], mcnemar: float,
          t0_mcnemar: Optional[float] = None) -> str:
    r"""顶会规范:
    †  = 95% CI 不跨 0 (VIB vs Full)
    ‡  = 配对 McNemar 检验 p<0.05 —— 且 temperature=0 下仍 p<0.05 (稳健显著)
    §  = 采样下 McNemar p<0.05 但贪心解码下 p≥0.05 (边缘显著, 对温度敏感)
    """
    stars = ""
    lo, hi = ci
    if not (lo <= 0 <= hi):
        stars += r"$^\dagger$"
    if mcnemar < 0.05:
        if t0_mcnemar is not None and t0_mcnemar >= 0.05:
            stars += r"$^\S$"
        else:
            stars += r"$^\ddagger$"
    return stars


# ---------------------------------------------------------------
# Table 1: Vanilla / Full / VIB × HumanEval · MBPP · GSM8K
# ---------------------------------------------------------------

TABLE1_HEADER = r"""\begin{table*}[t]
\centering
\caption{Pass@1 main results: Vanilla single-agent baseline vs Full-Comm 3-agent vs VIB-Sparse (ours).
Point estimates (pp) and 95\% paired-bootstrap CIs are reported for VIB vs Full; VIB vs Vanilla reports the raw pp difference without CI for brevity.
$\dagger$: 95\% CI does not span zero;
$\ddagger$: McNemar $p<0.05$ under both sampling and greedy (temperature$=0$, robust);
$\S$: sampling $p<0.05$ but greedy $p\ge 0.05$ (marginal, temperature-sensitive).
MBPP compression is near zero because max\_steps$=2$ caps each side at $\sim 6$ messages saturating discussion; MATH follows the same frozen-history Planner-Solver-Verifier pipeline as GSM8K.
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
Vanilla is single-agent direct synthesis (no team discussion, temperature$=0.1$, max\_tokens$=1024$), sharing synthesis hyperparameters with Full/VIB to keep apples-to-apples comparison.
MBPP message compression is near zero because the 3-agent max\_steps$=2$ bounds each side to $\approx 6$ messages saturating the discussion budget; VIB savings concentrate on GSM8K ($+30.49\%$), MATH ($+25.59\%$) and HumanEval ($+21.14\%$).
\textbf{Temperature robustness (greedy reruns):}
message compression under temperature$=0$ barely changes: MATH 25.59$\to$25.39\% ($\Delta=-0.20$pp), GSM8K 30.49$\to$30.28\% ($\Delta=-0.21$pp) --- VIB compression is not a sampling artefact.
\textbf{MATH significance envelope:}
sampling regression $-0.92$pp (McNemar $p=0.041$), greedy $-0.72$pp ($p=0.132$); magnitude stays $-0.7\sim-0.9$pp but significance is marginal across temperatures, see $\S$ marker.}
\end{table*}
"""


def render_table1(rows: List[DatasetRow]) -> str:
    lines = [TABLE1_HEADER]
    for r in rows:
        star = _star(r.delta_vib_full, r.delta_vib_full_ci, r.mcnemar_p, r.t0_mcnemar_p)
        lines.append(
            f"{r.label:<15} & {r.total:>5} & "
            f"{_pct(r.vanilla_pass):>7} & "
            f"{_pct(r.full_pass)}~{_fmt_ci(*r.full_wilson):>14} & "
            f"{_pct(r.vib_pass)}~{_fmt_ci(*r.vib_wilson):>14} & "
            f"{_pp(r.delta_vib_vanilla):>6} & "
            f"{_pp(r.delta_vib_full)}{star:>6} & {_fmt_ci(*r.delta_vib_full_ci):>14} & "
            f"{_pp(r.msg_compress_vib_full):>6} & "
            f"{_pp(r.token_save_vib_full):>6} \\\\"
        )
    lines.append(TABLE1_FOOTER)
    return "\n".join(lines)


# ---------------------------------------------------------------
# Table 2: MBPP Early-Stopped vs Multi-Round
# ---------------------------------------------------------------

TABLE2_HEADER = r"""\begin{table}[t]
\centering
\caption{MBPP internal Early-Stopped / Multi-Round split.
ES = early-stopped when the 1st-round probe synthesis passes immediately; MR = multi-round discussion required.
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
        f"Early-Stopped & {mbpp.es_count} ({_pct(mbpp.es_pct)}\\%) & "
        f"{_pct(mbpp.es_vanilla):>6} & "
        f"{_pct(mbpp.es_full):>6} & "
        f"{_pct(mbpp.es_vib):>6} & "
        f"{_pp(round(mbpp.es_vib - mbpp.es_full, 2)):>6} & "
        f"{_pp(0.0):>6} \\\\"
    )
    lines.append(
        f"Multi-Round    & {mbpp.mr_count} ({_pct(mbpp.mr_pct)}\\%) & "
        f"{_pct(mbpp.mr_vanilla):>6} & "
        f"{_pct(mbpp.mr_full):>6} & "
        f"{_pct(mbpp.mr_vib):>6} & "
        f"{_pp(mbpp.mr_delta_vib_full):>6} & "
        f"{_pp(1.59):>6} \\\\"
    )
    lines.append(TABLE2_FOOTER)
    return "\n".join(lines)


# ---------------------------------------------------------------
# Table 3: GSM8K Ablation: frozen-only vs frozen+role-weights
# ---------------------------------------------------------------

TABLE3_HEADER = r"""\begin{table}[t]
\centering
\caption{GSM8K ablation on $N=200$ sub-sample toggling role-aware weights (Section 3, Method).
Frozen-history-only VIB already achieves $+30.25\%$ message compression;
adding role-aware weights (frozen+role) still holds $+25.17\%$ compression while Full baseline drops 1~pp --- VIB is robust to both history regimes.
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
temperature $0.1$), so the backbone family is the only variable. VIB's compression and its non-inferior accuracy
transfer from DeepSeek v4-pro to Qwen-3.8-Max: message reduction stays above $25\%$ and accuracy does not drop for either family.
\label{tab:crossmodel}}"""

TABLE5_NOTE = r"""\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{\sloppy\footnotesize
\textbf{Reading the token column.} VIB compresses the \emph{shared context}, not the model's output:
prompt tokens fall by $4.6$--$5.2\%$, but steering both backbones on a shorter history makes them emit a
\emph{longer} chain-of-thought, so completion tokens rise ($+0.45\%$ DeepSeek, $+3.64\%$ Qwen) and the
\emph{net} token saving ($1.2$--$1.7\%$) trails the message/prompt reduction.
The mechanism is therefore best read as \textbf{context compression}; the end-to-end token saving is real
but bounded by how much the backbone lengthens its own reasoning once the context shrinks.}"""


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
configuration, so their agreement is a direct run-to-run noise bound at this $N$.
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
\textbf{Interpretation.} VIB accuracy is insensitive to all three scalars: across the %d grid points
the VIB$-$Full $\Delta$ deviates from the default by at most $%.2f$ pp, so the headline result is not
an artefact of a lucky hyperparameter choice. Message reduction is governed by the two scalars that
set how aggressively the gate trades sufficiency against redundancy: $\beta_0$ spans
$%.1f$\%%--$%.1f$\%% (%s as it grows) and $\gamma$ spans $%.1f$\%%--$%.1f$\%% (%s), whereas
$\lambda$ shifts it only over $%.1f$\%%--$%.1f$\%% across a threefold range. Because accuracy is
flat throughout, either $\beta_0$ or $\gamma$ serves as a deployment dial for the sparsity/cost
budget; we keep the defaults $\beta_0=3.2$, $\lambda=0.08$, $\gamma=-0.4$ used in all main results
and leave per-budget retuning to future work. The three bolded default rows are the \emph{same}
configuration and agree to within $%.2f$ pp in $\Delta$ and $%.2f$ pp in message reduction --- a
direct run-to-run noise bound at $N=200$.}""" % (
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


def _paired_vs_anchor(anchor: Dict[str, bool], arm: Dict[str, bool],
                      n_boot: int = 10000, seed: int = 20260812):
    """Paired bootstrap CI for (arm − anchor) in pp over the shared task ids.

    Reuses the manuscript's single bootstrap implementation: the helper consumes
    ``full_passed``/``vib_passed`` keys, so the anchor is mapped onto the former
    and the comparison arm onto the latter.
    """
    shared = [t for t in anchor if t in arm]
    if not shared:
        return (0.0, 0.0)
    rows = [{"full_passed": anchor[t], "vib_passed": arm[t]} for t in shared]
    return an._paired_delta_pp_bootstrap_ci(rows, n_boot=n_boot, seed=seed)


def render_table6_gating(sweep: Dict[str, Any], gating_dir: Path) -> str:
    rows = sweep["rows"]
    by_arm = {r["arm"]: r for r in rows}
    anchor = by_arm["vib"]

    anchor_json = load(gating_dir / f"{anchor['name']}.json")
    anchor_vib = {r["task_id"]: bool(r["vib_passed"]) for r in _detailed_rows(anchor_json)}
    anchor_full = {r["task_id"]: bool(r["full_passed"]) for r in _detailed_rows(anchor_json)}
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
        arm_map = {d["task_id"]: bool(d["vib_passed"]) for d in _detailed_rows(point)}
        label = naive_labels.get(r["arm"]) or f"Diversity top-$k$ ($\\theta={float(r['knob']):.2f}$)"
        # Use the same shared-task convention as the paired CI below, so the point
        # estimate and its interval cannot disagree about which tasks they summarise.
        shared = [t for t in anchor_vib if t in arm_map]
        delta = 100.0 * sum(arm_map[t] - anchor_vib[t] for t in shared) / max(1, len(shared))
        body_rows.append((
            label,
            float(r["summary"]["msg_compression_pct"]),
            float(r["summary"]["token_saving_pct"]),
            float(r["summary"]["vib_pass_rate"]),
            delta,
            _paired_vs_anchor(anchor_vib, arm_map),
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
\caption{Is the VIB utility signal doing the work? All arms share one backbone snapshot
(\texttt{__MODEL__}), one $200$-problem GSM8K subset and the identical harness, so only the
admission policy varies. \emph{Random drop} keeps each candidate with probability $p$ calibrated
to the VIB anchor's own admission rate, and \emph{diversity top-$k$} drops any candidate whose
maximum cosine against the already-broadcast history exceeds $\theta$ --- it reuses the same frozen
encoder but has no goal term, no $\beta_t$ schedule, no role prior and no guards.
$\Delta$ is the paired (arm$-$VIB) Pass@1 difference in pp with a $10\,000$-resample bootstrap 95\% CI;
negative means the arm is worse than VIB. The arms pinned to the anchor's message budget --- the random
drop and the nearest top-$k$ --- realise a reduction within $__BUDGET__$~pp of the anchor's; the
remaining top-$k$ settings remove strictly less and only trace the budget/accuracy trade-off.
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
        star = _star(r.delta_vib_full, r.delta_vib_full_ci, r.mcnemar_p, r.t0_mcnemar_p)
        print(f"| {r.label:<15} | {r.total:>5} | {r.vanilla_pass:>7.2f}% | "
              f"{r.full_pass:>7.2f}% {_fmt_ci(*r.full_wilson):>10} | "
              f"{r.vib_pass:>7.2f}% {_fmt_ci(*r.vib_wilson):>10} | "
              f"{r.delta_vib_vanilla:>+6.2f} | "
              f"{r.delta_vib_full:>+6.2f}{star:>4} | "
              f"{_fmt_ci(*r.delta_vib_full_ci):>18} | "
              f"{r.msg_compress_vib_full:>+6.2f} | {r.token_save_vib_full:>+6.2f} |")
    print("=" * len(hdr))
    # Print t0 robustness summary
    any_t0 = any(r.t0_mcnemar_p is not None for r in rows)
    if any_t0:
        print("\n── t0 (temperature=0) 稳健性诊断 ──")
        for r in rows:
            if r.t0_mcnemar_p is None:
                continue
            delta_rng = (min(r.delta_vib_full, r.t0_delta_vib_full),
                         max(r.delta_vib_full, r.t0_delta_vib_full))
            p_rng = (min(r.mcnemar_p, r.t0_mcnemar_p),
                     max(r.mcnemar_p, r.t0_mcnemar_p))
            comp_delta = (r.t0_msg_compress - r.msg_compress_vib_full
                          if r.t0_msg_compress is not None else None)
            sig = _star(r.delta_vib_full, r.delta_vib_full_ci, r.mcnemar_p, r.t0_mcnemar_p)
            print(f"  · {r.label:<12} Δ(VIB−Full) 采样={r.delta_vib_full:+.2f}pp / t0={r.t0_delta_vib_full:+.2f}pp  "
                  f"range=[{delta_rng[0]:+.2f},{delta_rng[1]:+.2f}]  "
                  f"McNemar p range=[{p_rng[0]:.4f},{p_rng[1]:.4f}]  "
                  f"marker={sig if sig else '(无)'}  "
                  f"compress Δ(t0−sampling)={'—' if comp_delta is None else f'{comp_delta:+.2f}pp'}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Export VIB-Sparse paper LaTeX tables from result JSONs")
    p.add_argument("--vanilla", default="results/vanilla_with_math_merged.json")
    p.add_argument("--main", default="results/main_with_math_merged.json")
    p.add_argument("--ablfrozen", default="results/gsm8k_ablation_frozen_only.json")
    p.add_argument("--ablrole", default="results/gsm8k_ablation_frozen_plus_role.json")
    p.add_argument("--outdir", default="paper/tables")
    p.add_argument("--t0-math", default="results/math_main_5000_t0_v2.json",
                   help="MATH temperature=0 重跑结果 JSON (含 summary.math)，用于稳健显著性标注")
    p.add_argument("--t0-gsm8k", default="results/gsm8k_main_1319_t0.json",
                   help="GSM8K temperature=0 重跑结果 JSON (含 summary.gsm8k)，用于稳健显著性标注")
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

    rows = [
        _build_row("humaneval", "HumanEval", van, main, _t0_for("humaneval")),
        _build_row("mbpp", "MBPP", van, main, _t0_for("mbpp")),
        _build_row("gsm8k", "GSM8K", van, main, _t0_for("gsm8k")),
    ]
    if "math" in main.get("summary", {}):
        rows.append(_build_row("math", "MATH", van, main, _t0_for("math")))

    t1 = render_table1(rows)
    (outdir / "table1_main_three_methods.tex").write_text(t1)
    (outdir / "table2_mbpp_es_mr_split.tex").write_text(render_table2(rows[1]))
    (outdir / "table3_gsm8k_ablation.tex").write_text(render_table3(abl1, abl2))
    (outdir / "_README.txt").write_text(README)
    written = ["table1_main_three_methods.tex", "table2_mbpp_es_mr_split.tex",
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

    preview_table1_plain(rows)
    print(f"\n✅ 已生成 {len(written)} 张 .tex + 1 README to {outdir.resolve()}:")
    for name in written:
        f = outdir / name
        print("   ·", f.name, f"({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

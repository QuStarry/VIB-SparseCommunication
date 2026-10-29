"""Export an offline report from validated experiment analyses, without API calls."""

import argparse
from html import escape
import json
from pathlib import Path

from src.request_ledger import fingerprint


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("full", "vib", "random_matched", "cosine_matched")
NAMES = {"full": "Full", "vib": "VIB", "random_matched": "随机匹配", "cosine_matched": "余弦匹配"}


def load_analysis(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = {key: value for key, value in result.items() if key != "content_sha256"}
    if "content_sha256" in result and fingerprint(payload) != result["content_sha256"]:
        raise ValueError("Analysis content fingerprint differs")
    return result


def render_report(feedback, cohorts):
    """Render factual charts with zero-based scales and explicit cost scopes."""
    primary = feedback["primary_comparison"]
    actual = feedback["actual_experiment_cost"]
    rows, data = [], []
    for policy in POLICIES:
        group = feedback["statistics"]["groups"][policy]
        communication = feedback["communication"][policy]
        costs = feedback["policy_costs"][policy]
        record = {
            "name": NAMES[policy], "policy": policy,
            "quality": group["task_mean_accuracy_pct"],
            "messages": communication["messages"],
            "cost": float(costs["standalone_one_synthesis_proxy_usd"]),
            "input": costs["round2"]["prompt_tokens"] + costs["synthesis"]["prompt_tokens"] / 2,
        }
        data.append(record)
        rows.append(
            f'<tr><th scope="row">{NAMES[policy]}</th>'
            f'<td>{group["passed_generations"]}/32</td><td>{record["quality"]:.4f}%</td>'
            f'<td>{communication["messages"]}/96</td>'
            f'<td>{record["input"]:,.1f}</td>'
            f'<td>${escape(costs["standalone_one_synthesis_proxy_usd"])}</td></tr>')
    prior_rows = []
    for title, tasks, analysis in cohorts:
        groups = analysis["statistics"]["groups"]
        values = " · ".join(f'{escape(name)} {group["task_mean_accuracy_pct"]:.2f}%'
                            for name, group in groups.items())
        prior_rows.append(f'<tr><th scope="row">{escape(title)}</th><td>{tasks}</td><td>{values}</td></tr>')
    lo, hi = primary["ci95_pp"]
    chart_data = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return TEMPLATE.replace("{{ROWS}}", "\n".join(rows)).replace("{{DATA}}", chart_data).replace(
        "{{PRIORS}}", "\n".join(prior_rows)).replace("{{COST}}", escape(actual["proxy_usd"])).replace(
        "{{TOKENS}}", f'{actual["prompt_tokens"] + actual["completion_tokens"]:,}').replace(
        "{{DELTA}}", f'{primary["delta_pp"]:+.4f}').replace("{{CI}}", f'[{lo:+.4f}, {hi:+.4f}]').replace(
        "{{P}}", f'{primary["exact_sign_p"]:.6f}').replace(
        "{{DIRECTIONS}}", f'{primary["left_higher_tasks"]} / {primary["right_higher_tasks"]} / {primary["tied_tasks"]}').replace(
        "{{TRUNCATED}}", escape(json.dumps(feedback["truncated_by_phase"], ensure_ascii=False)))


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>稀疏通信 · 前瞻实验报告</title>
<style>
:root{--paper:#f4f1e9;--ink:#202622;--muted:#5e655e;--line:#cdcfc5;--accent:#17654b}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 "PingFang SC","Helvetica Neue",sans-serif}
a{color:var(--accent);text-underline-offset:4px}a:hover{color:var(--ink)}button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:4px}
header{border-bottom:1px solid var(--line);padding:18px 5vw;display:flex;justify-content:space-between;gap:20px;font-size:13px}
main{max-width:1200px;margin:auto;padding:52px 5vw 70px}.eyebrow{font-size:12px;letter-spacing:.16em;color:var(--accent);text-transform:uppercase}
h1{font:normal clamp(34px,5vw,62px)/1.2 Georgia,"PingFang SC",serif;letter-spacing:-.03em;margin:16px 0 22px}
h2{font-size:23px;font-weight:500;margin:0 0 12px}p{max-width:860px}.muted{color:var(--muted)}
.lead{max-width:820px;font-size:18px}.summary{display:flex;gap:45px;padding:22px 0 30px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.summary strong{display:block;font:normal 30px/1.5 Georgia,serif;font-variant-numeric:tabular-nums}
.summary span{font-size:13px;color:var(--muted)}section{padding:36px 0;border-bottom:1px solid var(--line)}
.split{display:grid;grid-template-columns:1fr 1fr;gap:45px}.controls{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0}
button{font:inherit;border:1px solid var(--line);border-radius:0;padding:7px 15px;background:transparent;color:var(--ink);cursor:pointer;transition:background .15s,color .15s}
button[aria-pressed=true],button:hover{background:var(--ink);color:var(--paper)}.chart{margin:25px 0}
.bar-row{display:grid;grid-template-columns:100px minmax(80px,1fr) 105px;align-items:center;gap:15px;margin:18px 0}
.track{background:#e3e3d8;height:20px}.bar{height:100%;background:#858d80;transition:width .35s}.bar.vib{background:var(--accent)}
.number{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}
th,td{padding:12px 16px 12px 0;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}thead th{font-weight:500;color:var(--muted)}
.note{border-left:3px solid var(--accent);padding:2px 0 2px 18px;margin:22px 0}.flow{display:flex;gap:0;flex-wrap:wrap;margin:25px 0}
.flow div{flex:1;min-width:150px;padding:15px 18px 15px 0;border-top:2px solid var(--ink)}.flow small{display:block;color:var(--muted)}
details{padding:12px 0}summary{cursor:pointer;font-weight:500}.links{display:grid;grid-template-columns:1fr 1fr;gap:12px}.links a{padding:8px 0;border-bottom:1px solid var(--line)}
footer{padding-top:28px;font-size:13px;color:var(--muted)}code{font-size:.9em;overflow-wrap:anywhere}html{scroll-behavior:smooth}
@media(max-width:680px){main{padding-top:30px}.split,.links{grid-template-columns:1fr}.bar-row{grid-template-columns:80px 1fr 85px;font-size:13px;gap:8px}header{flex-wrap:wrap}.summary{gap:22px}table{font-size:12px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
@media print{button{display:none}main{padding:0}header{padding:10px 0}section{break-inside:avoid}body{background:white}}
</style></head><body>
<header><span>VIB / SPARSE COMMUNICATION</span><span>屈占祥 · 独立研究者 · 2026-10-28</span></header>
<main><div class="eyebrow">Prospective Evidence / 04</div><h1>两轮真实反馈实验</h1>
<p class="lead">让第二轮 Agent 真正读取各策略接收的第一轮历史，再比较最终质量、消息量与费用。历史主表不改，新批次单独报告。</p>
<div class="summary"><div><strong>16题 / 368次</strong><span>固定任务 · 全部真实请求 · 不补生成</span></div>
<div><strong>{{TOKENS}}</strong><span>实际请求总Token</span></div><div><strong>${{COST}}</strong><span>真实用量 × 历史代理价，非实付invoice</span></div></div>
<section aria-labelledby="chart-title"><h2 id="chart-title">四策略实测对照</h2><p class="muted">每题两次最终合成先取均值；统计单位为16题，不是32条输出，也不是两次团队轨迹。</p>
<div class="controls" role="group" aria-label="切换图表指标">
<button type="button" data-metric="quality" aria-pressed="true">准确率</button><button type="button" data-metric="messages" aria-pressed="false">接收消息</button>
<button type="button" data-metric="cost" aria-pressed="false">单策略代理成本</button><button type="button" data-metric="input" aria-pressed="false">后续输入Token</button></div>
<div id="chart" class="chart" aria-label="四策略指标柱状图"></div><p id="chart-note" class="muted" aria-live="polite"></p>
<div class="scroll"><table><caption style="text-align:left;padding:12px 0">精确数值与成本口径</caption><thead><tr><th>策略</th><th>成功/输出</th><th>任务均值准确率</th><th>接收消息</th><th>后续输入Token¹</th><th>单策略一次代理费用²</th></tr></thead><tbody>{{ROWS}}</tbody></table></div>
<p class="muted">¹ 本策略第二轮输入 + 两次合成输入的均值，不含共享第一轮。² 完整共享第一轮 + 本策略第二轮 + 一次合成平均费用。不是额外实付记录；实际实验总账包含四策略、两次合成，共享第一轮仅计一次。</p></section>
<section><div class="split"><div><h2>预声明主比较</h2><p>VIB − 随机数量匹配：<strong>{{DELTA}} pp</strong><br>任务bootstrap 95%区间：{{CI}}<br>双侧精确方向检验：p = {{P}}<br>VIB较高 / 随机较高 / 平局：{{DIRECTIONS}}题</p></div>
<div><h2>统计解释</h2><p>10000次任务bootstrap；方向检验只分析非平局题，两者不是同一统计量。其他比较探索性未校正。</p><p class="note">跨零不表示等效。全零任务差值时，[0,0]是样本bootstrap退化，不是总体无风险证明。没有预声明非劣margin。</p></div></div></section>
<section><h2>真实反馈与证据链</h2><div class="flow"><div>01 / 共享首轮<small>48次新候选</small></div><div>02 / 策略历史<small>真实编码与逐条门控</small></div><div>03 / 第二轮<small>192次按各组历史新生成</small></div><div>04 / 最终合成<small>128次，之后才隐藏评分</small></div></div>
<p>Architect、Coder、Reviewer同读轮前历史。VIB增量门控；随机与余弦每轮在自己的候选中选k条，k来自VIB本轮自产候选的通过数。因此匹配消息数，不匹配Token、角色构成或实付费用。</p>
<details><summary>编码、截断与恢复约束</summary><p>本地 pinned MiniLM，CPU/eval，384维，max_seq_length 256；长消息会按tokenizer截断。逐条向量和决策保存。候选截断保留付费非空原文，最终截断按失败计费；不修复或补生成。本批截断计数：<code>{{TRUNCATED}}</code>。</p><p>pending、raw、终态账目按顺序fsync；单次请求，SDK无自动重试。unknown/missing usage保守停机，不当作零。父manifest、report和route指纹连接阶段；已收到输出不重发。</p></details></section>
<section><h2>此前批次，分开解读</h2><div class="scroll"><table><thead><tr><th>批次</th><th>题数</th><th>各组任务均值准确率</th></tr></thead><tbody>{{PRIORS}}</tbody></table></div>
<p>24题简洁提示优势未在48题复验中复现；输入成本下降不等于精度等效。32题接收端四组质量相同，VIB少接收32.81%消息、少用17.27%合成输入，但旧候选生成成本已发生，不是端到端收益。</p></section>
<section><h2>能支持什么，尚缺什么</h2><p>保留此前真实历史正收益；新实验分别检验提示、接收端和两轮反馈。现有VIB包括多种代码启发式，不将所有效果归于单独的信息瓶颈公式。</p>
<p>本批仅16题、两轮、单团队轨迹、固定随机子集。同源已审计HumanEval不是新数据集，无法排除预训练污染；返回模型ID不是官方不可变快照认证。</p>
<p class="note">所有美元数值均为历史代理价：输入$0.30/百万Token、输出$1.20/百万Token。阶段观察预算$0.15/$0.45/$0.40，无实付invoice；字节规划不是verified Token上界，不保证人民币实付硬上限。</p>
<p>下一层关键证据：新数据、多团队和多随机种子、预声明功效或非劣界限、真实账单及等费用控制。当前适合严谨实证研究，不凭小样本宣称顶会级方法突破。</p></section>
<section><h2>一键证据导航</h2><nav class="links" aria-label="实验文件导航">
<a href="README.md">可引用综合说明</a><a href="analysis.json">逐题评分与全部统计</a>
<a href="protocol.json">冻结协议、源码与预算</a><a href="pipeline_report.json">阶段账目与父证据指纹</a>
<a href="routes_round1.json">第一轮向量与接收历史</a><a href="routes_round2.json">第二轮向量与接收历史</a>
<a href="../../paper/prospective_experiments_zh.md">中文论文补充</a><a href="../../paper/prospective_experiments_en.md">English supplement</a>
<a href="../../docs/feedback_experiment.md">执行与恢复方法</a><a href="../../paper/gap1_claim_audit_trail.md">历史事实与审计轨迹</a></nav></section>
<footer>离线自包含报告 · 无外部字体、图片、脚本或数据请求 · 图表从零起算 · Zhanxiang Qu / ORCID 0009-0000-0137-9751</footer>
</main><script>
const data={{DATA}};
const settings={
quality:{max:100,format:v=>v.toFixed(2)+'%',note:'横轴从0到100%。每题两次最终输出先平均，没有best-of-two。'},
messages:{max:96,format:v=>v+' 条',note:'横轴从0到96条。每题唯一接收历史，不把两次最终合成算成双倍消息。'},
cost:{max:Math.max(...data.map(d=>d.cost))*1.1,format:v=>'$'+v.toFixed(6),note:'横轴从0起算。完整共享首轮 + 本策略第二轮 + 一次最终合成平均费用；不是实际实验总支出。'},
input:{max:Math.max(...data.map(d=>d.input))*1.1,format:v=>v.toLocaleString('zh-CN'),note:'横轴从0起算。第二轮输入 + 一次最终合成平均输入；不包含共享第一轮。'}
};
function draw(metric){const s=settings[metric];document.querySelectorAll('[data-metric]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.metric===metric)));
document.getElementById('chart').innerHTML=data.map(d=>'<div class="bar-row"><span>'+d.name+'</span><div class="track"><div class="bar '+d.policy+'" style="width:'+(100*d[metric]/s.max)+'%"></div></div><span class="number">'+s.format(d[metric])+'</span></div>').join('');
document.getElementById('chart-note').textContent=s.note;}
document.querySelectorAll('[data-metric]').forEach(b=>b.addEventListener('click',()=>draw(b.dataset.metric)));draw('quality');
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/feedback_experiment_20261028/report.html")
    args = parser.parse_args()
    from experiments.analyze_feedback_experiment import analyze_feedback
    feedback = analyze_feedback(ROOT / "results/feedback_experiment_20261028")
    cohorts = [(title, count, load_analysis(ROOT / "results" / folder / "analysis.json"))
               for title, count, folder in (
                   ("提示探索", 24, "prompt_contrast_pilot_20261028"),
                   ("提示复验", 48, "prompt_contrast_replication_20261028"),
                   ("接收端对照", 32, "gate_receiver_20261028"))]
    html = render_report(feedback, cohorts)
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != html:
            raise ValueError("Existing report differs; use a new output")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(html)
    print(args.output)


if __name__ == "__main__":
    main()

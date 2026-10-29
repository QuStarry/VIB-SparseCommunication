"""Produce offline dual-scoring reports from verified frozen-output analysis."""

import argparse
import ast
from html import escape
import json
from pathlib import Path

from experiments.audit_plus_reference_suite import audit_reference_suite
from experiments.rescore_feedback_plus import load_dataset, rescore_feedback_plus
from experiments.run_feedback_experiment import immutable_json
from experiments.run_prompt_contrast_pilot import ROOT


POLICIES = ("full", "vib", "random_matched", "cosine_matched")
NAMES = {"full": "Full", "vib": "VIB", "random_matched": "随机匹配", "cosine_matched": "余弦匹配"}


def coverage_counts(tasks):
    """Count literal exported input arrays without executing test code."""
    counts = {}
    for task_id, task in tasks.items():
        check = next(node for node in ast.parse(task["test"]).body
                     if isinstance(node, ast.FunctionDef) and node.name == "check")
        assignments = {target.id: node.value for node in check.body if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        inputs = assignments.get("inputs")
        if not isinstance(inputs, (ast.List, ast.Tuple)):
            raise ValueError("Exported test input count is not statically available")
        counts[task_id] = len(inputs.elts)
    return counts


def render(result, manifest, counts):
    original = result["original_statistics"]["groups"]
    enhanced = result["statistics"]["groups"]
    primary = result["primary_comparison"]
    lo, hi = primary["ci95_pp"]
    table, chart = [], []
    for policy in POLICIES:
        old, new = original[policy], enhanced[policy]
        transitions = result["transitions"][policy]
        table.append(f'<tr><th scope="row">{NAMES[policy]}</th><td>{old["passed_generations"]}/32</td>'
                     f'<td>{new["passed_generations"]}/32</td><td>{new["task_mean_accuracy_pct"]:.4f}%</td>'
                     f'<td>{transitions["pass_to_fail"]}</td><td>{transitions["fail_to_pass"]}</td></tr>')
        chart.append({"name": NAMES[policy], "policy": policy,
                      "original": old["task_mean_accuracy_pct"], "enhanced": new["task_mean_accuracy_pct"]})
    task_rows, markdown_rows = [], []
    changed_tasks = []
    for task in manifest["task_ids"]:
        rows = [row for row in result["evaluations"] if row["task_id"] == task]
        changed = any(row["original_passed"] != row["passed"] for row in rows)
        if changed:
            changed_tasks.append(task)
        values = [sum(row["passed"] for row in rows if row["prompt_variant"] == policy) for policy in POLICIES]
        task_rows.append(f'<tr data-changed="{str(changed).lower()}"><th scope="row">{escape(task)}</th>'
                         f'<td>{counts[task]}</td>' + "".join(f"<td>{value}/2</td>" for value in values) + "</tr>")
        markdown_rows.append(f"| {task} | {counts[task]} | " + " | ".join(f"{value}/2" for value in values) + " |")
    conclusion = ("增强测试发现了原通过输出的新失败，应保留双口径并按逐题证据解释。"
                  if changed_tasks else "增强测试未改变本批标签：精度天花板仍存在，不证明全输入域正确或策略等效。")
    selected_count = sum(counts[task] for task in manifest["task_ids"])
    data = json.dumps(chart, ensure_ascii=False).replace("<", "\\u003c")
    page = HTML.replace("{{ROWS}}", "\n".join(table)).replace("{{TASKS}}", "\n".join(task_rows)).replace(
        "{{DATA}}", data).replace("{{INPUTS}}", f"{selected_count:,}").replace(
        "{{CONCLUSION}}", conclusion).replace("{{DELTA}}", f'{primary["delta_pp"]:+.4f}').replace(
        "{{CI}}", f"[{lo:+.4f}, {hi:+.4f}]").replace("{{P}}", f'{primary["exact_sign_p"]:.6f}')
    lines = [
        "# 冻结反馈输出增强重评", "", conclusion, "",
        "只重评上一轮固定16题、四策略、各两份最终输出，共128份。没有重新生成、修复、选优或按评分删题。"
        "原生成报告、通过标签、usage和历史代理成本不修改。", "",
        "## 双口径结果", "",
        "| 策略 | 原通过/32 | 增强通过/32 | 增强任务均值 | 原通过→失败 | 原失败→通过 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        old, new, change = original[policy], enhanced[policy], result["transitions"][policy]
        lines.append(f'| {NAMES[policy]} | {old["passed_generations"]} | {new["passed_generations"]} | '
                     f'{new["task_mean_accuracy_pct"]:.4f}% | {change["pass_to_fail"]} | {change["fail_to_pass"]} |')
    lines.extend([
        "", f'VIB减随机主比较：{primary["delta_pp"]:+.4f}pp；10000次任务bootstrap95%区间'
        f'[{lo:+.4f}, {hi:+.4f}]；双侧精确方向p={primary["exact_sign_p"]:.6f}。'
        "统计单位仍为16题，不是128份输出或输入组数；零差时区间退化，不表示等效。",
        "", "这是生成结束后增加的覆盖审计，不是新预注册确认性实验，也不是新数据集。"
        "主比较沿用原方案，其余探索性未校正；没有预声明非劣margin。",
        "", "## 数据与验收", "",
        f"发布方`{manifest['dataset']['publisher']}`，固定修订`{manifest['dataset']['revision']}`。"
        f"镜像下载SHA-256与发布方LFS对象一致：`{manifest['dataset']['verified_publisher_lfs_sha256']}`。",
        "", f"导出测试共164题、{sum(counts.values()):,}组输入；本16题{selected_count:,}组。"
        "全部164个发布方参考实现各执行两次通过，共328次参考执行；随后128份原输出各执行两次，共256次候选执行。",
        "", "使用固定HF导出`check(candidate)`，没有认证GitHub release v0.1.10，也没有声称运行官方EvalPlus CLI。"
        "整体每进程30秒超时，与官方逐输入参考时间缩放不同。显式绑定NumPy依赖目录，`-I -S`独立解释器不继承凭证，"
        "两次状态不一致为设施错误。这不是恶意代码网络或文件系统安全沙箱。",
        "", "发布方声明Apache-2.0许可，原始题面/参考实现/测试以固定数据文件保留，不将参考实现用作生成或修复材料。",
        "", "## 逐题增强评分", "",
        "| 任务 | 增强输入组数 | Full | VIB | 随机 | 余弦 |", "|---|---:|---:|---:|---:|---:|",
        *markdown_rows, "", "## 成本与研究判断", "",
        "新增模型请求0，新增模型代理费用$0；原368次生成的历史代理费用$0.0908907保持不变。"
        "没有新增invoice证据。增加的是本地参考和候选评分工作。",
        f'本地参考评分累计墙钟{result["local_reference_evaluation_seconds"]:.3f}秒；'
        f'候选评分累计墙钟{result["local_candidate_evaluation_seconds"]:.3f}秒。'
        "这是本机实测各记录时间之和，不是跨设备性能基准，也不包含完整预检和报告导出。",
        "", "同源题目即使增强测试通过，也不能排除预训练污染或证明全域语义正确。"
        "单团队轨迹、固定随机子集和16题功效限制仍在；不能据双口径满分宣称VIB独有优势或顶会方法突破。"
        "下一实验应改进新数据与多团队/多随机子集可识别性，而非按本批评分修复或反复调参。",
        "", "## 一键导航", "",
        "| 内容 | 入口 |", "|---|---|",
        "| 离线双口径报告 | [report.html](report.html) |",
        "| 原文数据与来源 | [publisher_test.jsonl](publisher_test.jsonl) |",
        "| 协议与旧证据指纹 | [manifest.json](manifest.json) |",
        "| 164题参考验收 | [reference_acceptance.json](reference_acceptance.json) |",
        "| 逐份原/增强标签 | [analysis.json](analysis.json) |",
        "| 独立参考记录 | [reference_checks/](reference_checks/) |",
        "| 独立候选记录 | [candidate_checks/](candidate_checks/) |",
        "| 原结果保持不变 | [原反馈报告](../feedback_experiment_20261028/README.md) |",
        "| 隔离与恢复方法 | [操作说明](../../docs/enhanced_rescoring.md) |", "",
    ])
    return page, "\n".join(lines), changed_tasks


HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>反馈输出 · 增强测试重评</title><style>
:root{--ink:#242a27;--muted:#59665e;--paper:#f4f1e9;--line:#cdd2c9;--accent:#17654b}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.7 "PingFang SC","Helvetica Neue",sans-serif}
header{padding:18px 5vw;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:15px;font-size:13px}
main{max-width:1160px;margin:auto;padding:36px 5vw 65px}h1{font:normal clamp(30px,4vw,52px)/1.3 Georgia,"PingFang SC",serif;margin:15px 0}
h2{font-size:23px;font-weight:500}p{max-width:870px}.muted{color:var(--muted)}section{border-top:1px solid var(--line);padding:20px 0;margin-top:25px}
.metrics{display:flex;gap:45px;flex-wrap:wrap;margin:25px 0}.metrics strong{display:block;font:normal 30px Georgia,serif}.metrics span{font-size:13px;color:var(--muted)}
button{font:inherit;cursor:pointer;padding:8px 16px;border:1px solid var(--line);background:transparent;color:var(--ink);transition:background .15s}
button[aria-pressed=true],button:hover{background:var(--ink);color:var(--paper)}button:focus-visible,a:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
.controls{display:flex;gap:8px;flex-wrap:wrap}.chart-row{display:grid;grid-template-columns:90px 1fr 100px;gap:15px;align-items:center;margin:18px 0}
.track{height:20px;background:#e0e4da}.bar{height:100%;background:#858e80;transition:width .3s}.bar.vib{background:var(--accent)}
.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}
th,td{padding:11px 15px 11px 0;border-bottom:1px solid var(--line);text-align:right}th:first-child{text-align:left}thead th{font-weight:500;color:var(--muted)}
.note{padding-left:18px;border-left:3px solid var(--accent)}a{color:var(--accent);text-underline-offset:4px}.links{display:flex;gap:20px;flex-wrap:wrap}
code{overflow-wrap:anywhere;font-size:.85em}footer{font-size:13px;color:var(--muted);margin-top:30px}
@media(max-width:650px){header{flex-wrap:wrap}.metrics{gap:25px}.chart-row{grid-template-columns:75px 1fr 85px;gap:8px}table{font-size:12px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}@media print{button{display:none}main{padding:10px}body{background:white}}
</style></head><body><header><span>VIB / COVERAGE AUDIT</span><span>Zhanxiang Qu · 2026-10-29</span></header><main>
<p class="muted">冻结输出 · 双轨评分 · 零新增模型请求</p><h1>增强测试重评</h1><p>{{CONCLUSION}}</p>
<div class="metrics"><div><strong>16题 / 128份</strong><span>同一批冻结最终输出</span></div><div><strong>{{INPUTS}}</strong><span>本批增强输入组数，非独立样本量</span></div><div><strong>0次 / $0</strong><span>新增模型请求与代理费用</span></div></div>
<section><h2>原评分与增强评分</h2><div class="controls" role="group" aria-label="评分口径">
<button type="button" data-track="original" aria-pressed="false">原评分</button><button type="button" data-track="enhanced" aria-pressed="true">增强评分</button></div>
<div id="chart" aria-label="四策略准确率"></div><p class="muted">横轴从0到100%。每题两次最终输出先平均；原评分记录不改写。</p>
<div class="scroll"><table><thead><tr><th>策略</th><th>原通过/32</th><th>增强通过/32</th><th>增强均值</th><th>通过→失败</th><th>失败→通过</th></tr></thead><tbody>{{ROWS}}</tbody></table></div>
<p class="note">VIB − 随机：{{DELTA}} pp；任务bootstrap95%区间{{CI}}；精确方向p={{P}}。全零差值会使区间退化，不能据此证明总体无损或等效。</p></section>
<section><h2>逐题增强结果</h2><div class="controls"><button type="button" id="changed" aria-pressed="false">只看标签变化题</button><span id="filter-status" class="muted" aria-live="polite"></span></div>
<div class="scroll"><table><thead><tr><th>任务</th><th>输入组数</th><th>Full</th><th>VIB</th><th>随机</th><th>余弦</th></tr></thead><tbody id="task-table">{{TASKS}}</tbody></table></div>
<p class="muted">每格为两份最终输出通过数，不是best-of-two。输入组数不改变16题的统计独立单位。</p></section>
<section><h2>来源与有效性</h2><p>发布方 evalplus/humanevalplus 的固定HF修订，镜像数据SHA-256与发布方LFS对象完全一致。164个发布方参考实现全部通过两次独立检查后，才对128份冻结输出重评。</p>
<p>使用HF导出check(candidate)，不是已认证GitHub release v0.1.10或官方EvalPlus CLI。每进程整体30秒超时，不等于官方逐输入参考时间缩放。</p>
<p class="note">这是同源题目的覆盖审计，不是新数据集或独立确认。参考实现只用于验收；不重新生成、修复、选优或按失败删题。预训练污染、单团队轨迹、固定随机子集与功效限制仍存在。</p>
<p>隔离解释器不继承模型凭证；显式加入NumPy依赖目录而不启用site初始化。两次不一致为设施错误。这不是恶意代码网络或文件系统安全沙箱。</p></section>
<section><h2>一键证据导航</h2><nav class="links"><a href="README.md">完整结果说明</a><a href="analysis.json">逐份双标签与统计</a><a href="manifest.json">来源及旧证据指纹</a><a href="reference_acceptance.json">164题参考验收</a><a href="../feedback_experiment_20261028/report.html">原报告，保持不变</a><a href="../../docs/enhanced_rescoring.md">执行与恢复</a></nav></section>
<footer>自包含离线报告 · 无外部字体、脚本、图片或数据请求 · 历史生成费用$0.0908907不变，非实付invoice</footer></main>
<script>const data={{DATA}};
function draw(track){document.querySelectorAll('[data-track]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.track===track)));
document.getElementById('chart').innerHTML=data.map(d=>'<div class="chart-row"><span>'+d.name+'</span><div class="track"><div class="bar '+d.policy+'" style="width:'+d[track]+'%"></div></div><span class="num">'+d[track].toFixed(2)+'%</span></div>').join('');}
document.querySelectorAll('[data-track]').forEach(b=>b.addEventListener('click',()=>draw(b.dataset.track)));draw('enhanced');
document.getElementById('changed').addEventListener('click',function(){const only=this.getAttribute('aria-pressed')!=='true';this.setAttribute('aria-pressed',String(only));let visible=0;document.querySelectorAll('#task-table tr').forEach(r=>{r.hidden=only&&r.dataset.changed!=='true';if(!r.hidden)visible++;});document.getElementById('filter-status').textContent=only?(visible?visible+'道题有标签变化':'没有标签变化题'):'显示全部16题';});
</script></body></html>"""


def export_report(output_dir):
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    counts = coverage_counts(load_dataset(manifest["dataset"]["path"]))
    if (output / "source_audit.json").exists():
        audit = audit_reference_suite(output)
        if not audit["all_passed"]:
            page, markdown = render_source_audit(audit, manifest, counts)
            changed = []
        else:
            page, markdown, changed = render(rescore_feedback_plus(output), manifest, counts)
    else:
        page, markdown, changed = render(rescore_feedback_plus(output), manifest, counts)
    for name, text in (("report.html", page), ("README.md", markdown)):
        path = output / name
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise ValueError("Existing enhanced report differs")
        else:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
    immutable_json(output / "coverage.json", {
        "input_groups_by_task": counts, "all_input_groups": sum(counts.values()),
        "selected_input_groups": sum(counts[task] for task in manifest["task_ids"]),
        "changed_task_ids": changed, "independent_unit": "16_tasks_not_input_groups",
        "candidate_scores_available": (output / "analysis.json").exists(),
    })
    return output / "report.html"


def render_source_audit(audit, manifest, counts):
    """Unavailable candidate scores are not rendered as zero accuracy."""
    failed = audit["failed_references"]
    task_rows = "\n".join(f'<tr><th scope="row">{escape(row["task_id"])}</th>'
                          f'<td>{escape(row["status"])}</td></tr>' for row in failed)
    passed = audit["passed_references"]
    selected_count = sum(counts[task] for task in manifest["task_ids"])
    header_style = HTML.split("<style>", 1)[1].split("</style>", 1)[0]
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>增强重评 · 来源验收未通过</title>
<style>{header_style}</style></head><body><header><span>VIB / SOURCE ACCEPTANCE</span>
<span>Zhanxiang Qu · 2026-10-29</span></header><main><p class="muted">增强评分源审计 · 原实验不变</p>
<h1>增强重评暂停</h1><p>发布方导出测试未通过参考验收，不能据此发布模型候选的增强准确率。</p>
<div class="metrics"><div><strong>{passed}/164</strong><span>参考实现通过</span></div>
<div><strong>128份未重评</strong><span>不是零分，也不是增强测试通过</span></div>
<div><strong>0次 / $0</strong><span>新增模型请求与代理费用</span></div></div>
<section><h2>参考验收失败</h2><div class="scroll"><table><thead><tr><th>任务</th><th>独立双检查状态</th></tr></thead>
<tbody>{task_rows}</tbody></table></div><p>164个参考实现各检查两次，共{audit["independent_reference_executions"]}次独立执行。
失败记录完整保留。候选评分守卫没有放行，不删题、不修改发布方测试，不修复或重新生成模型输出。</p></section>
<section><h2>已定位的导出断言问题</h2><p>HumanEval/32的原样断言为：</p>
<p><code>assert _poly(*candidate(*inp), inp) &lt;= 0.0001</code></p>
<p>find_zero应返回浮点根；断言先尝试将该浮点数用<code>*</code>展开。
独立复现得到<code>TypeError: Value after * must be an iterable, not float</code>。
这是评分源失败，不是VIB候选退化。未自行修改oracle后冒称官方成绩。</p></section>
<section><h2>两种口径的状态</h2><div class="scroll"><table><thead><tr><th>策略</th><th>原测试</th><th>增强测试</th></tr></thead><tbody>
{''.join(f'<tr><th scope="row">{NAMES[policy]}</th><td>32/32，100%</td><td>未执行</td></tr>' for policy in POLICIES)}
</tbody></table></div><p class="note">原100%记录保持不变；增强准确率不可用，不能把未执行填为0或100%。
未完成增强重评，也不能据此声称全输入域正确或策略等效。</p></section>
<section><h2>来源与范围</h2><p>固定HF修订<code>{manifest["dataset"]["revision"]}</code>，
下载镜像SHA-256与发布方LFS指纹一致。全库声明{sum(counts.values()):,}组输入，本16题声明{selected_count:,}组。
部分测试源失败，输入数量不是已通过覆盖，也不是独立统计样本。</p>
<p>使用发布方HF导出测试，不是已认证GitHub release v0.1.10或EvalPlus CLI。
整体每评分进程30秒超时，与官方逐输入参考时间缩放不同。
同源HumanEval增强测试不是新数据集，不消除污染、单团队轨迹和16题功效限制。</p></section>
<section><h2>一键证据导航</h2><nav class="links"><a href="README.md">完整审计说明</a>
<a href="source_audit.json">164题参考结果</a><a href="manifest.json">来源与原输出绑定</a>
<a href="publisher_test.jsonl">字节原样导出数据</a>
<a href="../feedback_experiment_20261028/report.html">原报告保持不变</a>
<a href="../../docs/enhanced_rescoring.md">隔离与守卫说明</a></nav></section>
<footer>离线自包含 · 无外部请求 · 原账目不变 · 候选增强成绩未发布</footer></main></body></html>"""
    lines = [
        "# 增强重评来源审计：候选评分暂停", "",
        f"完整参考源审计：{passed}/164参考实现通过，{len(failed)}题失败，"
        f'共{audit["independent_reference_executions"]}次独立解释器执行。'
        "**原128份候选增强评分未执行，不发布增强准确率。**", "",
        "## 失败证据", "", "| 任务 | 双检查状态 |", "|---|---|",
        *(f'| {row["task_id"]} | {row["status"]} |' for row in failed), "",
        "HumanEval/32原导出断言：", "", "```python",
        "assert _poly(*candidate(*inp), inp) <= 0.0001", "```", "",
        "find_zero的接口返回浮点根；上述断言将浮点数用`*`解包，独立复现得到"
        "`TypeError: Value after * must be an iterable, not float`。"
        "这是发布方导出oracle的错误，不是VIB候选失败。原数据字节不修改，未自行修补后声称官方成绩。", "",
        "原参考验收在HumanEval/32处按预声明守卫停止，未评分任何候选。随后独立源审计继续覆盖全部164题，"
        "只增加参考诊断，不移除失败题或授权候选评分。失败32不在原16题中，但不能事后改变已冻结的全库参考守卫以凑结果。", "",
        "## 原结果与增强结果状态", "",
        "| 策略 | 原反馈通过 | 增强通过 |", "|---|---:|---|",
        *(f"| {NAMES[policy]} | 32/32（100%） | 未执行 |" for policy in POLICIES), "",
        "原评分、代码、Token和历史代理费用保留，不将未执行写成零分或满分。"
        "目前无法回答更强测试下的候选准确率，亦不能将本次来源审计用于证明策略等效。", "",
        "## 来源、覆盖与成本", "",
        f'发布方`{manifest["dataset"]["publisher"]}`，修订`{manifest["dataset"]["revision"]}`。'
        f'镜像SHA-256与发布方LFS对象一致：`{manifest["dataset"]["verified_publisher_lfs_sha256"]}`。',
        f"全库声明{sum(counts.values()):,}组测试输入，本16题声明{selected_count:,}组。"
        "这是静态数组数量，不是通过覆盖或独立样本数。", "",
        "采用固定HF导出测试，不是已认证GitHub release v0.1.10，也不声称执行官方EvalPlus CLI。"
        "整体30秒超时不等于官方逐输入参考时间缩放。隔离解释器不继承凭证，"
        "显式绑定NumPy依赖目录；这不是恶意代码网络/文件系统沙箱。",
        f'本地参考评分记录累计时间{audit["local_reference_evaluation_seconds"]:.3f}秒，'
        "是本机实测记录之和，不是跨设备性能基准。", "",
        "新增模型请求0，新增模型代理费用$0；原368次生成历史代理费用$0.0908907不变，无新invoice证据。", "",
        "## 下一步判断", "",
        "优先获取并验证正确的官方输入型测试资产，或另行预声明独立测试协议，再做冻结输出重评。"
        "不修复模型、不按当前参考失败删题，不把事后放宽验收包装成确认性结果。"
        "跨数据集、多团队轨迹和等费用对照仍未完成，不能据本次工作宣称已验证。",
        "", "## 一键导航", "", "| 内容 | 入口 |", "|---|---|",
        "| 离线审计报告 | [report.html](report.html) |",
        "| 全164题参考源审计 | [source_audit.json](source_audit.json) |",
        "| 原样测试数据 | [publisher_test.jsonl](publisher_test.jsonl) |",
        "| 冻结来源及旧证据 | [manifest.json](manifest.json) |",
        "| 独立参考记录 | [reference_checks/](reference_checks/) |",
        "| 原反馈报告，不改写 | [原报告](../feedback_experiment_20261028/report.html) |",
        "| 方法与恢复 | [操作说明](../../docs/enhanced_rescoring.md) |", "",
    ]
    return page, "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/feedback_plus_rescore_20261029")
    print(export_report(parser.parse_args().output_dir))

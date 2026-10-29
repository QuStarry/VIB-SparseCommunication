"""Export explicitly local repaired scores without changing older reports."""

import argparse
from html import escape
import json
from pathlib import Path

from experiments.rescore_feedback_plus import load_dataset
from experiments.rescore_repaired_plus import PARENT, rescore_repaired
from experiments.run_feedback_experiment import immutable_json
from experiments.run_prompt_contrast_pilot import ROOT
from scripts.export_plus_rescore_report import coverage_counts, render


def render_local(result, manifest, counts):
    page, markdown, changed = render(result, manifest, counts)
    patch = manifest["patch"]
    note = (
        '<section><h2>本地修订口径</h2><p class="note">这是经用户确认的本地增强评分修订版，'
        '不是官方EvalPlus成绩。仅32题test字段修订，其余163题和本16题测试内容不变。</p>'
        '<p>修正标量根参数错误，使用精确有理数绝对残差1e-4；'
        '残差更大时仅接受相邻浮点值之间有精确零点符号跨越。没有任意扩大绝对阈值。</p>'
        '<p>原32题Newton参考确实存在不收敛：残差约−1.117亿。它的代码和失败记录保留；'
        '验收参考为163个原canonical + 32题发布方886组预期根表。不是164个原canonical全部通过。</p>'
        f'<p>派生数据SHA-256：<code>{escape(patch["local_file_sha256"])}</code>。</p>'
        '<nav class="links"><a href="patch.json">精确补丁与授权口径</a>'
        '<a href="original_task32_reference.json">原Newton失败记录</a>'
        '<a href="../feedback_plus_rescore_20261029/report.html">原来源失败报告</a></nav></section>'
    )
    page = page.replace("<h1>增强测试重评</h1>", "<h1>本地修订增强重评</h1>")
    page = page.replace("<title>反馈输出 · 增强测试重评</title>", "<title>反馈输出 · 本地修订增强重评</title>")
    page = page.replace("<section><h2>原评分与增强评分</h2>", note + "<section><h2>原评分与本地增强评分</h2>")
    page = page.replace("164个发布方参考实现全部通过两次独立检查后",
                        "163个原参考与32题发布方预期根表通过双检查后")
    page = page.replace("使用HF导出check(candidate)", "使用本地修订的HF导出check(candidate)")
    markdown = markdown.replace("# 冻结反馈输出增强重评", "# 本地修订增强重评")
    markdown = markdown.replace("全部164个发布方参考实现各执行两次通过",
                                "163个原canonical与32题发布方根表共164个验收参考各双检查通过")
    markdown = markdown.replace("使用固定HF导出", "使用本地修订的固定HF导出")
    markdown = markdown.replace("[publisher_test.jsonl](publisher_test.jsonl)",
                                "[本地派生数据](local_repaired_test.jsonl)")
    markdown += (
        "\n## 修订披露\n\n本报告不是官方EvalPlus成绩。仅32题test字段修改，输入和原canonical字段不改；"
        "32不在候选16题中，因此补丁只解除全库验收阻塞，并不直接修改本16题的测试内容。\n\n"
        "修订断言调用精确有理数残差检查，保留绝对1e-4；额外经用户确认，"
        "允许返回值相邻浮点数间存在精确符号跨越的零点。此数值接受范围已扩展，明确记录，不冒称完全原协议。\n\n"
        "原Newton在275组未收敛，残差约−1.117亿；原代码和失败诊断保留。用户确认仅验收参考换成"
        "发布方预期根表，886组根都必须接受同一数学oracle检查。其他163参考不改。\n\n"
        f"派生数据SHA-256：`{patch['local_file_sha256']}`。\n\n"
        "| 补充证据 | 入口 |\n|---|---|\n"
        "| 原数据和来源失败 | [原报告](../feedback_plus_rescore_20261029/report.html) |\n"
        "| 补丁及验收参考替换 | [patch.json](patch.json) |\n"
        "| 原Newton诊断 | [original_task32_reference.json](original_task32_reference.json) |\n"
        "| 原样官方数据 | [原文件](../feedback_plus_rescore_20261029/publisher_test.jsonl) |\n"
    )
    return page, markdown, changed


def export_local(output_dir):
    output = Path(output_dir)
    result = rescore_repaired(output)
    manifest = json.loads((output / "manifest.json").read_text())
    parent = json.loads((PARENT / "manifest.json").read_text())
    counts = coverage_counts(load_dataset(parent["dataset"]["path"]))
    page, markdown, changed = render_local(result, manifest, counts)
    for name, text in (("report.html", page), ("README.md", markdown)):
        path = output / name
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise ValueError("Existing local report changed")
        else:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
    immutable_json(output / "coverage.json", {
        "selected_input_groups": sum(counts[task] for task in manifest["task_ids"]),
        "changed_task_ids": changed, "scope": "local_repaired_scores",
        "task32_in_candidate_cohort": manifest["patch"]["task32_in_candidate_cohort"],
    })
    return output / "report.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/feedback_plus_repaired_20261029")
    print(export_local(parser.parse_args().output_dir))

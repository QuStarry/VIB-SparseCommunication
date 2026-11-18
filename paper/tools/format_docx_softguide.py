"""将投稿稿 .docx 按《软件导刊》通行排版规范套版。

规范要点（未在官方公开页面硬性给定的项，采用中文科技期刊通行值）:
- 正文：中文宋体、西文/数字 Times New Roman，五号(10.5pt)，行距固定 17 磅，
  首行缩进 2 字符，两端对齐。
- 标题：一级(文档标题)黑体三号(16pt)居中；二级(章)黑体小四(12pt)左顶格；
  三级(节)黑体五号(10.5pt)左顶格。
- 图题/表题：小五(9pt)宋体居中。
- 参考文献：小五(9pt)宋体，悬挂缩进。
- 表格：三线表（顶线/底线 1.5 磅，表头下横线 0.75 磅，无竖线、无内横线）。

用法: python3 paper/tools/format_docx_softguide.py <in.docx> [out.docx]
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION  # noqa: F401
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

CN_SONG = "宋体"
CN_HEI = "黑体"
EN_FONT = "Times New Roman"

BODY_PT = 10.5      # 五号
SMALL_PT = 9.0      # 小五
H2_PT = 12.0        # 小四
TITLE_PT = 16.0     # 三号
LINE_PT = 17.0      # 固定行距


def _set_rfonts(rpr, en: str, cn: str) -> None:
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)
    rfonts.set(qn("w:cs"), en)


def set_doc_defaults(doc: Document) -> None:
    styles = doc.styles.element
    dd = styles.find(qn("w:docDefaults"))
    if dd is None:
        return
    rpr = dd.find(qn("w:rPrDefault")).find(qn("w:rPr"))
    if rpr is not None:
        _set_rfonts(rpr, EN_FONT, CN_SONG)


def style_font(style, en: str, cn: str, size_pt: float, bold: bool | None = None) -> None:
    rpr = style.element.get_or_add_rPr()
    _set_rfonts(rpr, en, cn)
    if bold is not None:
        style.font.bold = bold
    style.font.size = Pt(size_pt)


def exact_line(p, pts: float) -> None:
    pf = p.paragraph_format
    pf.line_spacing = Pt(pts)
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY


def first_line_chars(p, chars: int = 2, size_pt: float = BODY_PT) -> None:
    """Word 原生「首行缩进 N 字符」，同时给出 twips 回退值。"""
    pPr = p._p.get_or_add_pPr()
    ind = pPr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        pPr.append(ind)
    ind.set(qn("w:firstLineChars"), str(chars * 100))
    ind.set(qn("w:firstLine"), str(int(size_pt * chars * 20)))


def clear_indent(p) -> None:
    pPr = p._p.get_or_add_pPr()
    ind = pPr.find(qn("w:ind"))
    if ind is not None:
        pPr.remove(ind)


def body_paragraph(p, size_pt: float = BODY_PT, indent: bool = True) -> None:
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    exact_line(p, LINE_PT)
    if indent:
        first_line_chars(p, 2, size_pt)
    else:
        clear_indent(p)
    for run in p.runs:
        rpr = run._r.get_or_add_rPr()
        _set_rfonts(rpr, EN_FONT, CN_SONG)
        run.font.size = Pt(size_pt)


def centered_paragraph(p, size_pt: float, bold: bool | None = None, cn: str = CN_SONG) -> None:
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    clear_indent(p)
    for run in p.runs:
        rpr = run._r.get_or_add_rPr()
        _set_rfonts(rpr, EN_FONT, cn)
        run.font.size = Pt(size_pt)
        if bold is not None:
            run.font.bold = bold


def ref_paragraph(p) -> None:
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.left_indent = Pt(2 * SMALL_PT)
    pf.first_line_indent = Pt(-2 * SMALL_PT)
    exact_line(p, SMALL_PT * 1.5)
    for run in p.runs:
        rpr = run._r.get_or_add_rPr()
        _set_rfonts(rpr, EN_FONT, CN_SONG)
        run.font.size = Pt(SMALL_PT)


def set_cell_borders(cell, *, top: bool = False, bottom: bool = False,
                     top_sz: int = 12, bottom_sz: int = 12) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcBorders")):
        tcPr.remove(old)
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        if edge == "top" and top:
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(top_sz))
        elif edge == "bottom" and bottom:
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(bottom_sz))
        else:
            e.set(qn("w:val"), "nil")
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "000000")
        borders.append(e)
    tcPr.append(borders)


def three_line_table(table) -> None:
    nrows = len(table.rows)
    for r, row in enumerate(table.rows):
        is_first = r == 0
        is_last = r == nrows - 1
        for cell in row.cells:
            # 顶线（首行顶）、表头下横线（首行底，细）、底线（末行底，粗）
            set_cell_borders(
                cell,
                top=is_first,
                bottom=is_first or is_last,
                top_sz=12,
                bottom_sz=6 if is_first else 12,
            )
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                clear_indent(p)
                for run in p.runs:
                    rpr = run._r.get_or_add_rPr()
                    _set_rfonts(rpr, EN_FONT, CN_SONG)
                    run.font.size = Pt(SMALL_PT)
                    if is_first:
                        run.font.bold = True


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法: format_docx_softguide.py <in.docx> [out.docx]")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src

    doc = Document(str(src))

    # 页面：A4 + 2.5cm 页边距
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    set_doc_defaults(doc)

    # 样式级：正文与标题
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if name in [s.name for s in doc.styles]:
            st = doc.styles[name]
            style_font(st, EN_FONT, CN_SONG, BODY_PT)
    if "Heading 1" in [s.name for s in doc.styles]:
        style_font(doc.styles["Heading 1"], EN_FONT, CN_HEI, TITLE_PT, bold=True)
    if "Heading 2" in [s.name for s in doc.styles]:
        style_font(doc.styles["Heading 2"], EN_FONT, CN_HEI, H2_PT, bold=True)
    if "Heading 3" in [s.name for s in doc.styles]:
        style_font(doc.styles["Heading 3"], EN_FONT, CN_HEI, BODY_PT, bold=True)

    # 段落级覆盖
    for p in doc.paragraphs:
        text = p.text.strip()
        name = p.style.name
        if name == "Heading 1":
            centered_paragraph(p, TITLE_PT, bold=True, cn=CN_HEI)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(12)
        elif name == "Heading 2":
            pf = p.paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.space_before = Pt(8)
            pf.space_after = Pt(4)
            clear_indent(p)
            for run in p.runs:
                rpr = run._r.get_or_add_rPr()
                _set_rfonts(rpr, EN_FONT, CN_HEI)
                run.font.size = Pt(H2_PT)
                run.font.bold = True
        elif name == "Heading 3":
            pf = p.paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.space_before = Pt(4)
            pf.space_after = Pt(2)
            clear_indent(p)
            for run in p.runs:
                rpr = run._r.get_or_add_rPr()
                _set_rfonts(rpr, EN_FONT, CN_HEI)
                run.font.size = Pt(BODY_PT)
                run.font.bold = True
        elif name == "Captioned Figure":
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            clear_indent(p)
        elif name == "Image Caption":
            centered_paragraph(p, SMALL_PT)
        elif text.startswith("作者：") or text.startswith("中图分类号"):
            centered_paragraph(p, BODY_PT)
        elif text[:1] == "表" and any(ch.isdigit() for ch in text[:6]):
            centered_paragraph(p, SMALL_PT)
        elif len(text) > 2 and text.startswith("[") and text[1].isdigit():
            ref_paragraph(p)
        else:
            body_paragraph(p)

    for table in doc.tables:
        three_line_table(table)

    doc.save(str(dst))
    print(f"saved: {dst}")


if __name__ == "__main__":
    main()

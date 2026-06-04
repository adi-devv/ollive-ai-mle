"""
Generate evaluation_report.pdf from evaluation_report.md
Run: python generate_pdf.py
"""

import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


MD_PATH = Path(__file__).parent / "evaluation_report.md"
PDF_PATH = Path(__file__).parent / "evaluation_report.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm


def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
                             fontSize=20, textColor=colors.HexColor("#1a1a2e"),
                             spaceAfter=12, spaceBefore=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                             fontSize=14, textColor=colors.HexColor("#16213e"),
                             spaceAfter=8, spaceBefore=14,
                             borderPad=4),
        "h3": ParagraphStyle("h3", parent=base["Heading3"],
                             fontSize=12, textColor=colors.HexColor("#0f3460"),
                             spaceAfter=6, spaceBefore=10),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontSize=10, leading=15,
                               textColor=colors.HexColor("#333333"),
                               spaceAfter=4),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"],
                                 fontSize=10, leading=14,
                                 leftIndent=16, bulletIndent=0,
                                 textColor=colors.HexColor("#333333"),
                                 spaceAfter=3),
        "code": ParagraphStyle("code", parent=base["Code"],
                               fontSize=9, leading=13,
                               backColor=colors.HexColor("#f4f4f4"),
                               textColor=colors.HexColor("#c7254e"),
                               leftIndent=8),
        "caption": ParagraphStyle("caption", parent=base["Normal"],
                                  fontSize=8, textColor=colors.grey,
                                  alignment=TA_CENTER, spaceAfter=6),
    }
    return styles


def md_to_story(md_text: str, styles: dict):
    story = []
    lines = md_text.splitlines()
    table_lines = []
    in_table = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Table detection
        if line.strip().startswith("|"):
            table_lines.append(line)
            i += 1
            continue
        else:
            if table_lines:
                story.append(render_table(table_lines, styles))
                story.append(Spacer(1, 8))
                table_lines = []

        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 6))
        elif stripped.startswith("### "):
            story.append(Paragraph(stripped[4:], styles["h3"]))
        elif stripped.startswith("## "):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#cccccc"),
                                    spaceAfter=4))
            story.append(Paragraph(stripped[3:], styles["h2"]))
        elif stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], styles["h1"]))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            text = inline_fmt(stripped[2:])
            story.append(Paragraph(f"• {text}", styles["bullet"]))
        elif re.match(r"^\d+\. ", stripped):
            text = inline_fmt(re.sub(r"^\d+\. ", "", stripped))
            story.append(Paragraph(f"• {text}", styles["bullet"]))
        elif stripped.startswith("> "):
            story.append(Paragraph(inline_fmt(stripped[2:]), styles["caption"]))
        elif stripped.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#dddddd"),
                                    spaceBefore=4, spaceAfter=4))
        else:
            story.append(Paragraph(inline_fmt(stripped), styles["body"]))

        i += 1

    if table_lines:
        story.append(render_table(table_lines, styles))

    return story


def inline_fmt(text: str) -> str:
    """Convert inline markdown to ReportLab markup."""
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # Inline code
    text = re.sub(r"`(.+?)`", r'<font name="Courier" size="9" color="#c7254e">\1</font>', text)
    # Links — show text only
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    # Escape ampersands not already escaped
    text = text.replace("&", "&amp;") if "&amp;" not in text else text
    return text


def render_table(table_lines: list, styles: dict):
    rows = []
    header_row = None
    for j, tl in enumerate(table_lines):
        if re.match(r"^\s*\|[-| :]+\|\s*$", tl):
            continue
        cells = [c.strip() for c in tl.strip().strip("|").split("|")]
        if header_row is None:
            header_row = j
            rows.append([Paragraph(f"<b>{inline_fmt(c)}</b>",
                                   ParagraphStyle("th", fontSize=9,
                                                  textColor=colors.white,
                                                  leading=12))
                         for c in cells])
        else:
            rows.append([Paragraph(inline_fmt(c),
                                   ParagraphStyle("td", fontSize=9, leading=12))
                         for c in cells])

    if not rows:
        return Spacer(1, 4)

    col_count = max(len(r) for r in rows)
    usable_w = PAGE_W - 2 * MARGIN
    col_w = usable_w / col_count

    t = Table(rows, colWidths=[col_w] * col_count, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f9f9f9"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    styles = build_styles()
    story = md_to_story(md_text, styles)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="AI Assistants Evaluation Report",
        author="Aadit Singal",
    )
    doc.build(story)
    print(f"PDF generated: {PDF_PATH}")


if __name__ == "__main__":
    main()

"""PDF report builder for completed MammoFM jobs."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Mapping, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_VIEW_ORDER = ("LCC", "LMLO", "RCC", "RMLO")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontSize=18,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=14,
            spaceAfter=8,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
        ),
        "report": ParagraphStyle(
            "Report",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=9,
            leading=11,
        ),
    }


def _meta_table(rows: list[tuple[str, str]]):
    tbl = Table(rows, colWidths=[1.2 * inch, 5.5 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def _thumbnail_grid(image_paths: Mapping[str, Path]):
    cell_w, cell_h = 3.1 * inch, 3.1 * inch
    rows = []
    pair = []
    for view in _VIEW_ORDER:
        path = image_paths.get(view) or image_paths.get(view.lower())
        if path and Path(path).exists():
            img = Image(str(path), width=cell_w, height=cell_h, kind="proportional")
            cell = [Paragraph(f"<b>{view}</b>", _styles()["meta"]), img]
        else:
            cell = [Paragraph(f"<b>{view}</b> (missing)", _styles()["meta"])]
        pair.append(cell)
        if len(pair) == 2:
            rows.append(pair)
            pair = []
    grid = Table(rows, colWidths=[cell_w + 0.1 * inch] * 2)
    grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return grid


def _report_paragraphs(text: str, style):
    text = (text or "").strip() or "(empty)"
    # reportlab Paragraph treats <br/> as a line break; escape angle brackets and convert newlines.
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\n", "<br/>")
    return Paragraph(safe, style)


def build_report_pdf(
    job_id: str,
    patient_id: Optional[str],
    exam_id: Optional[str],
    preliminary: str,
    final: str,
    image_paths: Mapping[str, Path],
    created_at: Optional[datetime] = None,
) -> bytes:
    """Render a 2-page PDF: header+thumbnails on page 1, both reports on page 2. Returns the PDF bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"MammoFM Report {job_id}",
    )
    s = _styles()
    ts = (created_at or datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S UTC")

    story = []
    story.append(Paragraph("MammoFM Report", s["title"]))
    story.append(
        _meta_table(
            [
                ("Patient ID:", patient_id or "—"),
                ("Exam ID:", exam_id or "—"),
                ("Job ID:", job_id),
                ("Generated:", ts),
            ]
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Input views", s["h2"]))
    story.append(_thumbnail_grid(image_paths))

    story.append(PageBreak())
    story.append(Paragraph("Preliminary Report (Stage 1)", s["h2"]))
    story.append(_report_paragraphs(preliminary, s["report"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Final Report (Stage 2)", s["h2"]))
    story.append(_report_paragraphs(final, s["report"]))

    doc.build(story)
    return buf.getvalue()

"""
mg_report.py
Generates a downloadable one-page PDF screening summary.

Uses reportlab (pure Python, no system dependencies):
    pip install reportlab

The PDF is the patient/clinician-facing artifact: the estimate, what drove it
for this specific patient, and the recommended follow-up -- with the research
disclaimer carried on the page itself so it cannot be separated from the result.
"""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)


DISCLAIMER = (
    "This document is produced by a research screening prototype. It is NOT a medical "
    "diagnosis and must not be used as the sole basis for any clinical decision. "
    "Myasthenia Gravis is confirmed by clinical examination and antibody testing "
    "(AChR / MuSK), not by this model."
)

_RISK_COLORS = {
    "High": colors.HexColor("#B23B14"),
    "Moderate": colors.HexColor("#95590A"),
    "Low": colors.HexColor("#0F6E56"),
}


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(
        name="TitleBig", parent=s["Title"], fontSize=17, spaceAfter=2, leading=21,
        textColor=colors.HexColor("#1F2937"),
    ))
    s.add(ParagraphStyle(
        name="Sub", parent=s["Normal"], fontSize=8.5, textColor=colors.HexColor("#6B7280"),
        spaceAfter=8,
    ))
    s.add(ParagraphStyle(
        name="H2", parent=s["Heading2"], fontSize=11.5, spaceBefore=11, spaceAfter=5,
        textColor=colors.HexColor("#1F2937"),
    ))
    s.add(ParagraphStyle(name="Body", parent=s["Normal"], fontSize=9.2, leading=13))
    s.add(ParagraphStyle(
        name="Small", parent=s["Normal"], fontSize=7.6, leading=10,
        textColor=colors.HexColor("#4B5563"),
    ))
    s.add(ParagraphStyle(
        name="Disclaim", parent=s["Normal"], fontSize=7.8, leading=10.5,
        textColor=colors.HexColor("#7F1D1D"),
    ))
    return s


def build_screening_pdf(
    probability,
    category,
    message,
    threshold,
    local_contributions=None,
    recommendations=None,
    patient_values=None,
    model_info=None,
    patient_label="Unnamed patient",
):
    """
    Build the PDF and return it as bytes (ready for a Streamlit download button).

    local_contributions : DataFrame from mg_explain.local_feature_contributions
    recommendations     : list of dicts from generate_local_recommendations
    model_info          : optional dict, e.g. {"Model": "...", "Test ROC-AUC": "..."}
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title="MG Screening Summary",
    )
    st = _styles()
    story = []

    story.append(Paragraph("Myasthenia Gravis -- Screening Summary", st["TitleBig"]))
    story.append(Paragraph(
        f"Research prototype output &nbsp;|&nbsp; Generated "
        f"{datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;|&nbsp; {patient_label}",
        st["Sub"],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#D1D5DB")))
    story.append(Spacer(1, 8))

    # --- Result block ---
    risk_color = _RISK_COLORS.get(category, colors.HexColor("#1F2937"))
    result_tbl = Table(
        [[
            Paragraph("<b>Estimated probability</b>", st["Body"]),
            Paragraph("<b>Screening category</b>", st["Body"]),
            Paragraph("<b>Threshold applied</b>", st["Body"]),
        ], [
            Paragraph(f"<font size=15><b>{probability:.1%}</b></font>", st["Body"]),
            Paragraph(f'<font size=15 color="{risk_color.hexval()}"><b>{category}</b></font>', st["Body"]),
            Paragraph(f"<font size=15><b>{threshold:.2f}</b></font>", st["Body"]),
        ]],
        colWidths=[58 * mm, 58 * mm, 62 * mm],
    )
    result_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#D1D5DB")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(result_tbl)
    story.append(Spacer(1, 6))
    story.append(Paragraph(message, st["Body"]))

    # --- Local explainability ---
    if local_contributions is not None and len(local_contributions) > 0:
        story.append(Paragraph("What drove this particular estimate", st["H2"]))
        story.append(Paragraph(
            "Each value below is measured by neutralising that single input and re-scoring "
            "this patient. It reflects this individual's result, not population averages.",
            st["Small"],
        ))
        story.append(Spacer(1, 4))

        up = local_contributions[local_contributions["Contribution"] > 1e-4].head(6)
        rows = [["Input", "Effect on probability", "Direction"]]
        for _, r in up.iterrows():
            rows.append([
                str(r["Feature"])[:44],
                f"{r['Contribution'] * 100:+.1f} pts",
                "raises risk",
            ])
        down = local_contributions[local_contributions["Contribution"] < -1e-4].tail(3)
        for _, r in down.iloc[::-1].iterrows():
            rows.append([
                str(r["Feature"])[:44],
                f"{r['Contribution'] * 100:+.1f} pts",
                "lowers risk",
            ])

        if len(rows) == 1:
            story.append(Paragraph(
                "No single input dominated this score -- it arises from the overall "
                "combination of values rather than one factor.", st["Body"],
            ))
        else:
            t = Table(rows, colWidths=[95 * mm, 43 * mm, 40 * mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D1D5DB")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t)

    # --- Recommendations ---
    if recommendations:
        story.append(Paragraph("Suggested follow-up, in order of contribution", st["H2"]))
        rec_rows = [["#", "Recommendation", "Basis"]]
        for r in recommendations:
            rec_rows.append([
                str(r["rank"]),
                Paragraph(r["recommendation"], st["Small"]),
                Paragraph(r.get("basis", ""), st["Small"]),
            ])
        rt = Table(rec_rows, colWidths=[8 * mm, 105 * mm, 65 * mm])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F6E56")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 8.2),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D1D5DB")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(rt)

    # --- Model provenance ---
    if model_info:
        story.append(Paragraph("Model details", st["H2"]))
        story.append(Paragraph(
            " &nbsp;|&nbsp; ".join(f"<b>{k}:</b> {v}" for k, v in model_info.items()),
            st["Small"],
        ))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#FCA5A5")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Important:</b> {DISCLAIMER}", st["Disclaim"]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

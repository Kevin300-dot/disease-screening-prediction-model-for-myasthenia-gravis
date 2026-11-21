"""Render a screening result (the dict returned by MGPredictor.predict) as a PDF report."""
from __future__ import annotations

import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

NAVY = colors.HexColor("#0B2545")
TEAL = colors.HexColor("#0E7C86")
INK = colors.HexColor("#1B2733")
MUTED = colors.HexColor("#5B6B7B")
LINE = colors.HexColor("#D5DCE4")
PANEL = colors.HexColor("#F3F6F9")
RED = colors.HexColor("#C0262D")
RED_BG = colors.HexColor("#FDECEC")
GREEN = colors.HexColor("#17794F")
GREEN_BG = colors.HexColor("#E7F6EE")
AMBER = colors.HexColor("#B26A00")
AMBER_BG = colors.HexColor("#FFF4DE")

PAGE_W, PAGE_H = A4
MARGIN = 15 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


def _style(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9, leading=12.5, textColor=INK)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "body": _style("body"),
    "small": _style("small", fontSize=8, leading=10.5, textColor=MUTED),
    "label": _style("label", fontSize=7, leading=9, textColor=MUTED, fontName="Helvetica-Bold"),
    "value": _style("value", fontSize=10, leading=13, fontName="Helvetica-Bold"),
    "h": _style("h", fontSize=10.5, leading=14, fontName="Helvetica-Bold", textColor=NAVY),
    "cell": _style("cell", fontSize=8.5, leading=11),
    "cellb": _style("cellb", fontSize=8.5, leading=11, fontName="Helvetica-Bold"),
    "th": _style("th", fontSize=7.5, leading=10, fontName="Helvetica-Bold", textColor=colors.white),
    "thc": _style("thc", fontSize=7.5, leading=10, fontName="Helvetica-Bold", textColor=colors.white,
                  alignment=TA_CENTER),
    "cellc": _style("cellc", fontSize=8.5, leading=11, alignment=TA_CENTER),
    "bullet": _style("bullet", leftIndent=10, bulletIndent=0, spaceAfter=2),
}


def P(text, style="body", **kw):
    st = S[style] if not kw else ParagraphStyle("tmp", parent=S[style], **kw)
    return Paragraph(escape(str(text)), st)


def _heading(text):
    t = Table([[Paragraph(escape(text.upper()), S["h"])]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _header(report):
    left = [Paragraph('<font color="white" size="18"><b>MyaScreen</b></font>', _style("hb", leading=22)),
            Paragraph('<font color="#BFD3E6" size="8.5">NEUROMUSCULAR SCREENING LABORATORY</font>',
                      _style("hs", leading=11))]
    right = [Paragraph('<font color="white" size="12"><b>Screening Report</b></font>',
                       _style("hr", alignment=TA_RIGHT, leading=15)),
             Paragraph(f'<font color="#BFD3E6" size="8">Report ID {escape(report["report_id"])}</font>',
                       _style("hr2", alignment=TA_RIGHT, leading=11)),
             Paragraph(f'<font color="#BFD3E6" size="8">{escape(report["generated_at"])}</font>',
                       _style("hr3", alignment=TA_RIGHT, leading=11))]
    t = Table([[left, right]], colWidths=[CONTENT_W * 0.55, CONTENT_W * 0.45])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW", (0, 0), (-1, -1), 3, TEAL),
    ]))
    return t


def _patient_table(report):
    p = report["patient"]
    weight = f'{p["weight_kg"]:g} kg' if p["weight_kg"] else "-"
    height = f'{p["height_cm"]:g} cm' if p["height_cm"] else "-"
    bmi = f'{p["bmi"]} ({p["bmi_category"]})' if p["bmi"] else "-"

    def cell(label, value):
        return [Paragraph(label.upper(), S["label"]), Paragraph(escape(str(value)), S["value"])]

    rows = [
        [cell("Patient name", p["name"]), cell("Age", f'{p["age"]} years'), cell("Sex", p["gender"])],
        [cell("Weight", weight), cell("Height", height), cell("Body mass index", bmi)],
        [cell("Symptom duration", p["symptom_duration"]), cell("Blood group", p.get("blood_group") or "-"),
         cell("Test requested", "Myasthenia gravis screening")],
    ]
    w = CONTENT_W / 3
    t = Table(rows, colWidths=[w * 1.25, w * 0.85, w * 0.9])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _result_box(res):
    positive = res["status"] == "POSITIVE"
    fg, bg = (RED, RED_BG) if positive else (GREEN, GREEN_BG)
    left = [
        Paragraph("MYASTHENIA GRAVIS", _style("rl", fontSize=11, leading=14, fontName="Helvetica-Bold",
                                              textColor=MUTED)),
        Paragraph(res["status"], _style("rs", fontSize=40, leading=44, fontName="Helvetica-Bold", textColor=fg)),
        Paragraph(f'Primary model: {escape(res["primary_model"])}  |  Algorithms returning positive: '
                  f'{res["models_positive"]}', _style("rm", fontSize=8, leading=11, textColor=MUTED)),
    ]
    right = [
        Paragraph("MODEL CONFIDENCE", _style("cl", fontSize=7.5, leading=10, fontName="Helvetica-Bold",
                                             textColor=MUTED, alignment=TA_CENTER)),
        Paragraph(f'{res["confidence_percent"]}%', _style("cv", fontSize=30, leading=34, fontName="Helvetica-Bold",
                                                         textColor=fg, alignment=TA_CENTER)),
        Paragraph(f'{res["confidence_band"]} confidence<br/>MG probability {res["mg_probability_percent"]}%',
                  _style("cb", fontSize=8, leading=11, textColor=MUTED, alignment=TA_CENTER)),
    ]
    t = Table([[left, right]], colWidths=[CONTENT_W * 0.62, CONTENT_W * 0.38])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 1.5, fg),
        ("LINEAFTER", (0, 0), (0, 0), 0.6, fg), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def _critical_box(text):
    t = Table([[Paragraph(f'<b>CRITICAL FINDING.</b> {escape(text)}',
                          _style("cr", fontSize=8.5, leading=12, textColor=RED))]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), RED_BG), ("BOX", (0, 0), (-1, -1), 0.8, RED),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _findings_table(findings):
    rows = [[Paragraph("CATEGORY", S["th"]), Paragraph("REPORTED BY PATIENT", S["th"])]]
    for sec in findings:
        items = sec["items"]
        text = "; ".join(items) if items else "None reported"
        style = "cell" if items else "small"
        rows.append([Paragraph(escape(sec["section"]), S["cellb"]), Paragraph(escape(text), S[style])])
    t = Table(rows, colWidths=[CONTENT_W * 0.24, CONTENT_W * 0.76], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _blood_table(blood):
    rows = [[Paragraph(h, S["th"] if i == 0 else S["thc"])
             for i, h in enumerate(["TEST", "RESULT", "UNIT", "REFERENCE RANGE", "FLAG"])]]
    for r in blood["panel"]:
        color = "#17794F" if r["status"] == "Normal" else "#C0262D"
        rows.append([Paragraph(escape(r["label"]), S["cell"]), Paragraph(f'<b>{r["value"]:g}</b>', S["cellc"]),
                     Paragraph(escape(r["unit"]), S["cellc"]), Paragraph(escape(r["reference"]), S["cellc"]),
                     Paragraph(f'<font color="{color}"><b>{r["status"].upper()}</b></font>', S["cellc"])])
    for a in blood["antibodies"]:
        if a["status"] == "Not tested":
            continue
        color = "#C0262D" if a["status"] == "Positive" else "#17794F"
        rows.append([Paragraph(escape(a["label"]), S["cell"]),
                     Paragraph(f'<font color="{color}"><b>{a["status"].upper()}</b></font>', S["cellc"]),
                     Paragraph("-", S["cellc"]), Paragraph("Negative", S["cellc"]), Paragraph("", S["cellc"])])
    t = Table(rows, colWidths=[CONTENT_W * 0.34, CONTENT_W * 0.14, CONTENT_W * 0.16, CONTENT_W * 0.2, CONTENT_W * 0.16],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _models_table(per_model):
    rows = [[Paragraph(h, S["th"] if i == 0 else S["thc"])
             for i, h in enumerate(["ALGORITHM", "RESULT", "MG PROBABILITY", "CONFIDENCE"])]]
    for m in per_model:
        pos = m["flags_mg"]
        name = m["model"] + ("  (primary)" if m["is_primary"] else "")
        rows.append([
            Paragraph(escape(name), S["cellb"] if m["is_primary"] else S["cell"]),
            Paragraph(f'<font color="{"#C0262D" if pos else "#17794F"}"><b>{m["result"]}</b></font>', S["cellc"]),
            Paragraph(f'{m["mg_probability_percent"]}%', S["cellc"]),
            Paragraph(f'{m["confidence_percent"]}%', S["cellc"]),
        ])
    t = Table(rows, colWidths=[CONTENT_W * 0.40, CONTENT_W * 0.2, CONTENT_W * 0.2, CONTENT_W * 0.2], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _performance_table(perf):
    heads = ["ALGORITHM", "ACCURACY", "PRECISION", "RECALL", "SPECIFICITY", "F1-SCORE", "ROC-AUC"]
    keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    rows = [[Paragraph(h, S["th"] if i == 0 else S["thc"]) for i, h in enumerate(heads)]]
    for name, m in perf["per_model"].items():
        primary = name == perf["primary_model"]
        rows.append([Paragraph(escape(name) + (" *" if primary else ""), S["cellb"] if primary else S["cell"])] +
                    [Paragraph(f"{m[k]}%", S["cellc"]) for k in keys])
    t = Table(rows, colWidths=[CONTENT_W * 0.27] + [CONTENT_W * 0.1217] * 6, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _bullets(items):
    return [Paragraph(escape(i), S["bullet"], bulletText="\u2022") for i in items]


def _specialist_table(specs):
    rows = [[Paragraph("SPECIALIST", S["th"]), Paragraph("WHEN", S["th"]), Paragraph("WHY", S["th"])]]
    for s in specs:
        rows.append([Paragraph(escape(s["specialist"]), S["cellb"]), Paragraph(escape(s["when"]), S["cell"]),
                     Paragraph(escape(s["why"]), S["cell"])])
    t = Table(rows, colWidths=[CONTENT_W * 0.30, CONTENT_W * 0.22, CONTENT_W * 0.48], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _hospital_table(care):
    """Nearest + best hospital for MG in Chennai (only when the patient gave a location)."""
    rows = [[Paragraph("", S["th"]), Paragraph("HOSPITAL", S["th"]), Paragraph("LOCATION", S["th"]),
             Paragraph("DISTANCE", S["thc"])]]
    entries = [("NEAREST", care["nearest"])]
    if not care["same_hospital"]:
        entries.append(("BEST FOR MG", care["best"]))
    else:
        entries[0] = ("NEAREST + BEST", care["nearest"])
    for tag, h in entries:
        why = h["evidence"][0]
        name = f'<b>{escape(h["name"])}</b><br/><font size="7.5" color="#5B6B7B">{escape(h["department"])}</font>'
        loc = escape(h["address"])
        if h.get("phone"):
            loc += f'<br/><font size="7.5" color="#5B6B7B">Phone: {escape(h["phone"])}</font>'
        rows.append([Paragraph(f"<b>{tag}</b>", S["cell"]), Paragraph(name, S["cell"]), Paragraph(loc, S["cell"]),
                     Paragraph(f'{h["distance_km"]} km', S["cellc"])])
        rows.append(["", Paragraph(f'<font size="7.5" color="#5B6B7B">Why: {escape(why)}</font>', S["cell"]), "", ""])
    t = Table(rows, colWidths=[CONTENT_W * 0.15, CONTENT_W * 0.33, CONTENT_W * 0.40, CONTENT_W * 0.12], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(2, len(rows), 2):
        style += [("SPAN", (1, i), (3, i)), ("LINEBELOW", (0, i), (-1, i), 0.4, LINE)]
    t.setStyle(TableStyle(style))
    return t


def build_report_pdf(data: dict) -> bytes:
    """`data` is the dict returned by MGPredictor.predict()."""
    report, res, adv = data["report"], data["result"], data["advice"]
    positive = res["status"] == "POSITIVE"
    buf = io.BytesIO()

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 10 * mm, f'MyaScreen  |  {report["patient"]["name"]}  |  {report["report_id"]}')
        canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                          bottomMargin=20 * mm, title=f'MyaScreen report - {report["patient"]["name"]}',
                          author="MyaScreen")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(MARGIN, 20 * mm, CONTENT_W, PAGE_H - MARGIN - 20 * mm,
                                                             leftPadding=0, rightPadding=0, topPadding=0,
                                                             bottomPadding=0)], onPage=on_page)])
    gap = lambda h=7: Spacer(1, h)
    story = [_header(report), gap(10), _patient_table(report), gap(10), _result_box(res), gap(8)]

    if adv.get("emergency"):
        story += [_critical_box(adv["emergency"]), gap(8)]

    story += [KeepTogether([_heading("Clinical interpretation"), gap(5), P(res["interpretation"])])]
    if res.get("clinical_pattern"):
        story += [gap(4), Paragraph(f'<b>Clinical pattern:</b> {escape(res["clinical_pattern"])}', S["body"])]
    if adv.get("age_sex_context"):
        story += [gap(3), P(adv["age_sex_context"], "small")]

    story += [gap(10), KeepTogether([_heading("Reported findings"), gap(5), _findings_table(data["findings"])])]
    blood = data.get("blood") or {}
    if blood.get("panel") or any(a["status"] != "Not tested" for a in blood.get("antibodies", [])):
        story += [gap(10), KeepTogether([_heading("Blood investigations"), gap(5), _blood_table(blood)])]
    if blood.get("group_note"):
        story += [gap(3), P(f'Blood group {blood["blood_group"]}: {blood["group_note"]}', "small")]
    story += [gap(10), KeepTogether([_heading("Screening algorithm results"), gap(5), _models_table(data["per_model"])])]

    perf = data["performance"]
    story += [gap(10), KeepTogether([
        _heading("Model performance"), gap(4),
        P(f'Evaluated on {perf["n_test"]:,} held-out patients not used for training '
          f'(* = primary model used for this report).', "small"), gap(4), _performance_table(perf)])]

    story += [gap(10), _heading("Recommendations"), gap(5), P("Next steps", "cellb")] + _bullets(adv["next_steps"])
    if adv.get("tests_to_ask_about"):
        story += [gap(5), P("Suggested confirmatory investigations", "cellb")] + _bullets(adv["tests_to_ask_about"])
    if adv.get("watch_for"):
        story += [gap(5), P("Warning signs to monitor", "cellb")] + _bullets(adv["watch_for"])
    if adv.get("daily_management"):
        story += [gap(5), P("Day-to-day management", "cellb")] + _bullets(adv["daily_management"])
    if adv.get("medication_safety"):
        story += [gap(5), P("Medication safety", "cellb"), P(adv["medication_safety"])]
    if adv.get("about_treatment"):
        story += [gap(5), P("About treatment", "cellb"), P(adv["about_treatment"])]

    story += [gap(10), KeepTogether([_heading("Recommended specialists"), gap(5),
                                     _specialist_table(data["recommended_specialists"])])]
    care = data.get("care_locator")
    if care:
        story += [gap(10), KeepTogether([
            _heading(f'Nearest and best hospital for myasthenia gravis ({care["region"]})'), gap(5),
            P(f'From {care["origin"]["label"]}. {care["summary"]}', "small"), gap(4), _hospital_table(care),
            gap(3), P(care["disclaimer"], "small")])]
    story += [gap(14), P(data["footer"], "small")]

    doc.build(story)
    return buf.getvalue()

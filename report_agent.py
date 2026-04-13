"""
Audit Report Generation Agent  (v3.0)
======================================
Changes from v2.2
-----------------
- Summarizer + Drafter merged into a SINGLE LLM call (saves 1 API call per run).
- Refiner is now OPTIONAL — disabled by default on free-tier to prevent quota
  exhaustion at the last step. Enable via ENABLE_REFINER=true env var.
- All nodes use gemini-2.5-flash (unchanged).
- Added PDF generation via ReportLab.

Pipeline:
  combined_drafter  →  [refiner?]  →  END
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import List, TypedDict

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
from agent import _invoke_with_retry, gemini_api_key

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import black, blue, grey, white, lightgrey
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT

load_dotenv()

# Set ENABLE_REFINER=true in .env to turn on the extra QA pass.
ENABLE_REFINER = os.getenv("ENABLE_REFINER", "false").lower() == "true"


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
class ReportState(TypedDict):
    raw_text:            str
    compliance_rules:    str
    calculation_results: str
    python_code:         str
    audit_findings:      List[str]
    audit_opinion:       str
    executive_summary:   str
    detailed_report:     str
    final_report:        str
    status:              str
    error:               str


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _today() -> str:
    return datetime.now().strftime("%B %d, %Y")


def _findings_block(findings: List[str]) -> str:
    return "\n\n".join(findings) if findings else "(no findings provided)"


def _flash_llm(max_tokens: int = 4096) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.05,
        google_api_key=gemini_api_key,
        max_output_tokens=max_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PDF Generation Output Logic
# ──────────────────────────────────────────────────────────────────────────────
def create_pdf_report(report_data: dict, output_path: str):
    """Converts the Markdown report output into a formal PDF."""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Custom Styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, spaceAfter=12, textColor=blue)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, spaceAfter=8, textColor=black)
    h3_style = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=12, spaceAfter=6, textColor=black)
    body_style = ParagraphStyle('BodyText', parent=styles['Normal'], fontSize=10, spaceAfter=6, alignment=TA_LEFT)

    # Title Page
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("INDEPENDENT AUDITOR'S REPORT", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"<b>Audit Opinion:</b> {report_data.get('audit_opinion', 'Unspecified')}", h2_style))
    elements.append(Paragraph(f"<b>Generated on:</b> {_today()}", body_style))
    elements.append(PageBreak())

    def format_md(text: str) -> str:
        # Translate simple Markdown to ReportLab-compatible HTML tags
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italics
        return text

    full_report = report_data.get('final_report', report_data.get('detailed_report', ''))
    in_code_block = False
    code_text = ""

    # Parse markdown line by line
    for line in full_report.split('\n'):
        if line.startswith('```'):
            if in_code_block:
                elements.append(Paragraph(f"<font name='Courier' size='8'>{code_text}</font>", body_style))
                code_text = ""
            in_code_block = not in_code_block
            continue

        if in_code_block:
            code_text += format_md(line) + "<br/>"
            continue

        stripped = line.strip()
        if stripped.startswith('### '):
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(format_md(stripped[4:]), h3_style))
        elif stripped.startswith('## '):
            elements.append(Spacer(1, 10))
            elements.append(Paragraph(format_md(stripped[3:]), h2_style))
        elif stripped.startswith('# '):
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(format_md(stripped[2:]), title_style))
        elif stripped.startswith('- ') or stripped.startswith('* '):
            elements.append(Paragraph(f"• {format_md(stripped[2:])}", body_style))
        elif stripped == '---':
            elements.append(Spacer(1, 10))
        elif stripped:
            elements.append(Paragraph(format_md(stripped), body_style))

    doc.build(elements)


# ──────────────────────────────────────────────────────────────────────────────
# Node 1 – Combined Drafter
# ──────────────────────────────────────────────────────────────────────────────
_DRAFT_SYSTEM = """\
You are a Big-4 audit partner producing a formal Independent Auditor's Report.

STEP 1 — Determine the audit opinion:
  • Unqualified  – financial statements present fairly in all material respects.
  • Qualified    – material misstatement(s) limited in scope/pervasiveness.
  • Adverse      – financial statements materially misstated on a pervasive basis.
  • Disclaimer   – unable to obtain sufficient appropriate audit evidence.

STEP 2 — Write the complete report in Markdown below this header:
  AUDIT OPINION: <type>
  ---
  <full report>

The report MUST contain ALL of these sections:

# INDEPENDENT AUDITOR'S REPORT
**Date:** {date}  **Audit Opinion:** <type>

---
## 1. Executive Summary
250–300 words. Opinion basis, top 2–3 findings with specific figures, risk posture.

---
## 2. Scope and Methodology
### 2.1 Scope
Documents reviewed, audit period, material scope limitations.
### 2.2 Methodology
Three-phase agentic workflow (ingestion → RAG analysis → report).
Applicable standards: ISA 700/705/706, CARO 2020, IFRS/IAS or GAAP.

---
## 3. Regulatory Compliance Assessment
Applicable CARO 2020 clauses retrieved and compliance status. Use a table.

---
## 4. Financial Analysis
### 4.1 Key Metrics Computed
Table: Metric | Value | Benchmark | Status
### 4.2 Trend Analysis
### 4.3 Anomaly Flags

---
## 5. Detailed Audit Findings
HIGH / MEDIUM / LOW groupings preserved from input.

---
## 6. Risk Assessment Matrix
| # | Finding | Severity | Likelihood | Impact | Priority |
One row per HIGH and MEDIUM finding.

---
## 7. Recommendations
Numbered. Each maps to a finding. Include: (a) action, (b) responsible party, (c) timeline.

---
## 8. Management Response Placeholder
_[To be completed by management within 30 days of report issuance.]_

---
## 9. Conclusion and Audit Opinion
Formal opinion with supporting rationale. Must match Section 1 opinion type.

---
## Appendix A – Python Calculation Code
Embed the Python code used for financial metric computation.

---
*Generated by Autonomous Financial Audit AI Framework v3.0.*

Keep total output under 2500 words. Cite specific rupee/percentage figures throughout.
"""

_DRAFT_USER = """\
Date: {date}

COMPLIANCE RULES & REGULATORY REQUIREMENTS:
{rules}

COMPUTED FINANCIAL METRICS:
{calculations}

DETAILED AUDIT FINDINGS:
{findings}

PYTHON CODE USED FOR CALCULATIONS:
```python
{python_code}
"""
async def drafter_node(state: ReportState) -> dict:
    """Single-shot: determine opinion + produce full report in one LLM call."""
    if state.get("status") == "failed":
        return state

    llm = _flash_llm(max_tokens=4096)
    user_prompt = _DRAFT_USER.format(
        date=_today(),
        rules=state["compliance_rules"][:2_500],
        calculations=state["calculation_results"],
        findings=_findings_block(state["audit_findings"]),
        python_code=state["python_code"][:1_500],
    )
    system_prompt = _DRAFT_SYSTEM.replace("{date}", _today())

    try:
        raw = await _invoke_with_retry([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
    except Exception as exc:
        return {"status": "failed", "error": f"Drafter failed: {exc}"}

    raw = raw.strip()

    opinion = "Qualified"
    report_body = raw
    if raw.upper().startswith("AUDIT OPINION:"):
        first_line, *rest = raw.splitlines()
        opinion = first_line.split(":", 1)[1].strip()
        report_body = "\n".join(rest).lstrip("-").strip()

    exec_summary = ""
    if "## 1. Executive Summary" in report_body:
        after = report_body.split("## 1. Executive Summary", 1)[1]
        exec_summary = after.split("---", 1)[0].strip()

    return {
        "audit_opinion":     opinion,
        "executive_summary": exec_summary,
        "detailed_report":   report_body,
        "status":            "draft_done",
    }
# ──────────────────────────────────────────────────────────────────────────────
# Node 2 – Refiner  (OPTIONAL — off by default on free tier)
# ──────────────────────────────────────────────────────────────────────────────
_REFINE_SYSTEM = """

You are a QA reviewer for a Big-4 audit firm. Review the draft and fix:
✓ Every HIGH/MEDIUM finding in Risk Matrix.
✓ Recommendations map to named findings.
✓ Figures in Executive Summary match Sections 4/5.
✓ Consistent severity labels (HIGH / MEDIUM / LOW only).
✓ Formal, unambiguous language.
✓ All 9 sections + appendix present and non-empty.
✓ Audit Opinion consistent across header and Section 9.
Output the complete corrected report only. No preamble. Under 2500 words.
"""

async def refiner_node(state: ReportState) -> dict:
    """QA pass — optional, disabled on free tier."""
    if state.get("status") == "failed":
        return state

    if not ENABLE_REFINER or not state.get("detailed_report"):
        return {"final_report": state.get("detailed_report", ""), "status": "report_complete"}

    llm = _flash_llm(max_tokens=4096)
    try:
        content = await _invoke_with_retry(llm, [
        SystemMessage(content=_REFINE_SYSTEM),
        HumanMessage(content=state["detailed_report"]),
    ])
    except Exception:
        content = state["detailed_report"]

    return {"final_report": content, "status": "report_complete"}
# ──────────────────────────────────────────────────────────────────────────────
# Passthrough when refiner is disabled
# ──────────────────────────────────────────────────────────────────────────────
async def passthrough_node(state: ReportState) -> dict:
    return {"final_report": state.get("detailed_report", ""), "status": "report_complete"}

# ──────────────────────────────────────────────────────────────────────────────
# Compile pipeline
# ──────────────────────────────────────────────────────────────────────────────
_wf = StateGraph(ReportState)
_wf.add_node("draft",  drafter_node)
_wf.add_node("refine", refiner_node if ENABLE_REFINER else passthrough_node)

_wf.set_entry_point("draft")
_wf.add_edge("draft",  "refine")
_wf.add_edge("refine", END)

report_pipeline = _wf.compile()
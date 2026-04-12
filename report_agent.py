"""
Step 3 – Audit Report Generation Agent
=========================================
Three-node LangGraph pipeline that turns structured audit findings into a
professional, publication-ready audit report.

  summarizer  →  drafter  →  refiner

FIX (v2.2): All nodes switched from gemini-2.5-pro to gemini-2.5-flash.
             Free-tier gemini-2.5-pro quota was 0 RPD causing RESOURCE_EXHAUSTED
             errors. gemini-2.5-flash has a far more generous free-tier allowance.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, TypedDict
import os

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
from agent import _invoke_with_retry, api_key


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


def _flash_llm(max_tokens: int = 2048) -> ChatGoogleGenerativeAI:
    """Shared factory — gemini-2.5-flash for all report nodes."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.05,
        google_api_key=api_key,
        max_output_tokens=max_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Node 1 – Summarizer
# ──────────────────────────────────────────────────────────────────────────────
_SUMMARY_SYSTEM = """\
You are a Big-4 audit partner preparing an executive summary for a board-level audience.

First, determine the appropriate audit opinion:
  • Unqualified  – financial statements present fairly in all material respects.
  • Qualified    – material misstatement(s) limited in scope or pervasiveness.
  • Adverse      – financial statements are materially misstated on a pervasive basis.
  • Disclaimer   – unable to obtain sufficient appropriate audit evidence.

Then write the executive summary (250–350 words) covering:
  1. Audit opinion type and basis
  2. Two or three most critical findings with specific figures cited
  3. Key financial health signals from the computed metrics
  4. Overall risk posture (Low / Moderate / High / Critical)

Format your response EXACTLY as:
  AUDIT OPINION: <type>
  ---
  <executive summary paragraphs>

Keep output under 500 words total.
"""

_SUMMARY_USER = """\
AUDIT FINDINGS:
{findings}

COMPUTED FINANCIAL METRICS:
{calculations}
"""


async def summarizer_node(state: ReportState) -> dict:
    """Determine audit opinion and draft executive summary."""
    if state.get("status") == "failed":
        return state

    # FIX: was gemini-2.5-pro — switched to flash (quota was exhausted)
    llm = _flash_llm(max_tokens=1024)

    try:
        raw = await _invoke_with_retry(llm, [
            SystemMessage(content=_SUMMARY_SYSTEM),
            HumanMessage(content=_SUMMARY_USER.format(
                findings=_findings_block(state["audit_findings"]),
                calculations=state["calculation_results"],
            )),
        ])
    except Exception as exc:
        return {"status": "failed", "error": f"Summarizer failed: {exc}"}

    raw          = raw.strip()
    opinion      = "Qualified"   # safe default
    summary_body = raw
    if raw.upper().startswith("AUDIT OPINION:"):
        first_line, *rest = raw.splitlines()
        opinion      = first_line.split(":", 1)[1].strip()
        summary_body = "\n".join(rest).lstrip("-").strip()

    return {
        "audit_opinion":     opinion,
        "executive_summary": summary_body,
        "status":            "summary_done",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 2 – Drafter
# ──────────────────────────────────────────────────────────────────────────────
_DRAFT_SYSTEM = """\
You are a senior audit manager drafting a formal Independent Auditor's Report.
Produce a complete, professional Markdown document.
Be specific and cite figures throughout.
Keep the total report under 2000 words.
"""

_DRAFT_USER = """\
Date: {date}
Audit Opinion: {opinion}

────────────── INPUTS ──────────────

EXECUTIVE SUMMARY:
{executive_summary}

COMPLIANCE RULES & REGULATORY REQUIREMENTS:
{rules}

COMPUTED FINANCIAL METRICS (Python analysis output):
{calculations}

DETAILED AUDIT FINDINGS:
{findings}

PYTHON CODE USED FOR CALCULATIONS:
```python
{python_code}
```

────────────── REQUIRED REPORT STRUCTURE ──────────────

# INDEPENDENT AUDITOR'S REPORT

**Date:** {date}
**Audit Opinion:** {opinion}

---

## 1. Executive Summary
<paste the executive summary here>

---

## 2. Scope and Methodology

### 2.1 Scope
Documents reviewed, audit period, and any material scope limitations.

### 2.2 Methodology
Three-phase agentic workflow:
  - Phase 1: Document ingestion and Knowledge Graph construction (LightRAG + Gemini)
  - Phase 2: RAG-based rule retrieval from pre-defined standards KB, Python-driven
    financial analysis, gap analysis
  - Phase 3: Automated report generation
Applicable standards (ISA 700/705/706, IFRS/IAS or GAAP as relevant).

---

## 3. Regulatory Compliance Assessment
Applicable rules retrieved and compliance status for each major requirement.
Use a table where helpful.

---

## 4. Financial Analysis
### 4.1 Key Metrics Computed
Present Python-derived metrics in a table: Metric | Value | Benchmark | Status

### 4.2 Trend Analysis
Period-over-period movements and their implications.

### 4.3 Anomaly Flags
Metrics outside acceptable ranges, with severity.

---

## 5. Detailed Audit Findings
Reproduce findings preserving HIGH / MEDIUM / LOW severity groupings.

---

## 6. Risk Assessment Matrix
| # | Finding | Severity | Likelihood | Impact | Priority |
|---|---------|----------|------------|--------|----------|
<one row per HIGH and MEDIUM finding>

---

## 7. Recommendations
Numbered. Map each to its finding. Include: (a) action, (b) responsible party, (c) timeline.

---

## 8. Management Response Placeholder
_[To be completed by management within 30 days of report issuance.]_

---

## 9. Conclusion and Audit Opinion
Formal audit opinion with supporting rationale.

---

## Appendix A – Python Calculation Code
Embed the Python code used for financial metric computation.

---

*This report was generated by the Autonomous Financial Audit AI Framework v2.2.*
"""


async def drafter_node(state: ReportState) -> dict:
    """Assemble all inputs into the full structured audit report."""
    if state.get("status") == "failed":
        return state

    llm = _flash_llm(max_tokens=4096)

    try:
        content = await _invoke_with_retry(llm, [
            SystemMessage(content=_DRAFT_SYSTEM),
            HumanMessage(content=_DRAFT_USER.format(
                date=_today(),
                opinion=state["audit_opinion"],
                executive_summary=state["executive_summary"],
                rules=state["compliance_rules"][:2_500],
                calculations=state["calculation_results"],
                findings=_findings_block(state["audit_findings"]),
                python_code=state["python_code"][:1_500],
            )),
        ])
    except Exception as exc:
        return {"status": "failed", "error": f"Drafter failed: {exc}"}

    return {
        "detailed_report": content,
        "status":          "draft_done",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 3 – Refiner
# ──────────────────────────────────────────────────────────────────────────────
_REFINE_SYSTEM = """\
You are a quality-assurance reviewer for a Big-4 audit firm.
Review the draft audit report and apply these checks:

  ✓ Every HIGH and MEDIUM finding appears in the Risk Assessment Matrix.
  ✓ Every Recommendation maps to a named finding.
  ✓ All figures quoted in the Executive Summary match figures in Section 4/5.
  ✓ Severity labels are consistent (HIGH / MEDIUM / LOW only).
  ✓ Language is formal, unambiguous, and free of hedging phrases.
  ✓ All nine numbered sections plus the appendix are present and non-empty.
  ✓ The Audit Opinion in the header matches Section 9.

Fix any issues found. Output the complete, corrected final report only.
Do not add any preamble or explanation outside the report itself.
Keep total output under 2500 words.
"""


async def refiner_node(state: ReportState) -> dict:
    """QA pass — produce the polished final report."""
    if state.get("status") == "failed":
        return state

    # FIX: was gemini-2.5-pro — switched to flash (quota was exhausted)
    llm = _flash_llm(max_tokens=4096)

    try:
        content = await _invoke_with_retry(llm, [
            SystemMessage(content=_REFINE_SYSTEM),
            HumanMessage(content=state["detailed_report"]),
        ])
    except Exception as exc:
        # Fallback: return the draft rather than failing the whole pipeline
        content = state["detailed_report"]

    return {
        "final_report": content,
        "status":       "report_complete",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Compile LangGraph pipeline
# ──────────────────────────────────────────────────────────────────────────────
_wf = StateGraph(ReportState)
_wf.add_node("summarize",    summarizer_node)
_wf.add_node("draft_report", drafter_node)
_wf.add_node("refine",       refiner_node)

_wf.set_entry_point("summarize")
_wf.add_edge("summarize",    "draft_report")
_wf.add_edge("draft_report", "refine")
_wf.add_edge("refine",       END)

report_pipeline = _wf.compile()
"""
Step 2 – RAG Analysis Agent  (v2.3 — CARO 2020 Integration)
=============================================================
Three-node LangGraph pipeline:

  librarian  →  calculator  →  auditor
      │               │            │
  Queries rules    LLM writes    CARO 2020 clause-by-clause
  ChromaDB +       Python code,  compliance check against
  doc raw text     executes it   rules + computed metrics

Changes in v2.3
---------------
- Auditor node now uses structured CARO 2020 clause data (all 21 clauses)
  for systematic compliance checking instead of generic gap analysis.
- LightRAG document queries replaced with direct raw-text analysis —
  the KG was returning "None" for all document queries in practice.
- Calculator prompt improved to actually print output.
- Rule queries remain concurrent via asyncio.gather.
"""

import ast
import io
import math
import statistics
import contextlib
import traceback
import asyncio
import os
from typing import TypedDict, List

import numpy as np
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent import (
    gemini_complete, gemini_embed,
    WORKING_DIR, EMBEDDING_DIM,
    _invoke_with_retry, _is_retryable,
    api_key,
)

from rules_store import query_rules

# CARO 2020 structured clause data
from caro_2020 import CARO_2020_CLAUSES, get_clause_summary


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
class AnalysisState(TypedDict):
    file_path:           str
    raw_text:            str
    compliance_rules:    str
    python_code:         str
    calculation_results: str
    audit_findings:      List[str]
    status:              str
    error:               str


# ──────────────────────────────────────────────────────────────────────────────
# Safe Python Interpreter
# ──────────────────────────────────────────────────────────────────────────────
_ALLOWED_BUILTINS = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "int", "isinstance",
    "iter", "len", "list", "map", "max", "min", "next", "pow",
    "print", "range", "round", "set", "sorted", "str", "sum",
    "tuple", "type", "zip",
    "None", "True", "False",
]


def _safe_globals() -> dict:
    import builtins as _b
    bdict = vars(_b)
    return {
        "__builtins__": {k: bdict[k] for k in _ALLOWED_BUILTINS if k in bdict},
        "math":         math,
        "statistics":   statistics,
        "np":           np,
    }


def safe_python_executor(code: str) -> dict:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {"success": False, "output": "", "error": f"SyntaxError: {exc}"}
    try:
        with (
            contextlib.redirect_stdout(stdout_buf),
            contextlib.redirect_stderr(stderr_buf),
        ):
            exec(code, _safe_globals())
        return {
            "success": True,
            "output":  stdout_buf.getvalue().strip(),
            "error":   stderr_buf.getvalue().strip(),
        }
    except Exception as exc:
        return {
            "success": False,
            "output":  stdout_buf.getvalue().strip(),
            "error":   f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# CARO helpers
# ──────────────────────────────────────────────────────────────────────────────
def _build_caro_checklist() -> str:
    """
    Compact but complete CARO 2020 checklist for prompt injection.
    Each clause: number, title, audit questions, key data fields.
    """
    lines = [
        "CARO 2020 — COMPANIES (AUDITOR'S REPORT) ORDER — ALL 21 CLAUSES",
        "=" * 65,
        "For each clause, assess: COMPLIANT / NON-COMPLIANT / INSUFFICIENT DATA / NOT APPLICABLE",
        "",
    ]
    for clause in CARO_2020_CLAUSES:
        lines.append(f"Clause {clause['clause_number']:02d}: {clause['title']}")
        lines.append(f"  Requirement: {clause['legal_text'][:200]}...")
        lines.append("  Check:")
        for q in clause["audit_questions"][:3]:   # top 3 questions per clause
            lines.append(f"    • {q}")
        lines.append(f"  Look for: {', '.join(clause['data_fields'][:4])}")
        lines.append("")
    return "\n".join(lines)


def _build_caro_data_fields() -> str:
    """All unique data fields across all 21 CARO clauses."""
    fields: set = set()
    for clause in CARO_2020_CLAUSES:
        fields.update(clause["data_fields"])
    return ", ".join(sorted(fields))


# ──────────────────────────────────────────────────────────────────────────────
# Node 1 – Librarian
# ──────────────────────────────────────────────────────────────────────────────
async def librarian_node(state: AnalysisState) -> dict:
    """
    Two-source retrieval:
    1. Rules ChromaDB  — pre-ingested regulatory standards
    2. Raw document text — direct excerpt (reliable; LightRAG KG returns None)
    """
    if state.get("status") == "failed":
        return state

    try:
        rule_queries = [
            "regulatory compliance requirements and applicable standards for financial audit",
            "financial reporting thresholds ratios capital adequacy benchmarks",
            "internal control requirements risk assessment and audit materiality criteria",
        ]

        rule_results = await asyncio.gather(
            *[query_rules(q, n_results=4) for q in rule_queries],
            return_exceptions=True,
        )

        rules_section = "## COMPLIANCE RULES & REGULATORY STANDARDS\n"
        rules_section += "(Retrieved from pre-ingested rules knowledge base)\n\n"
        for q, r in zip(rule_queries, rule_results):
            if isinstance(r, Exception):
                rules_section += f"[Query failed: {q}]\n"
            else:
                rules_section += f"### {q}\n{r}\n\n"

        # Use raw text directly — more reliable than LightRAG KG on free tier
        raw_excerpt = state["raw_text"][:8_000]
        doc_section = (
            "## DOCUMENT CONTENT (direct extract from uploaded financial PDF)\n\n"
            + raw_excerpt
        )

        combined = rules_section + "\n" + ("=" * 60) + "\n\n" + doc_section
        return {"compliance_rules": combined, "status": "rules_fetched"}

    except Exception as exc:
        return {"status": "failed", "error": f"Librarian error: {exc}"}


# ──────────────────────────────────────────────────────────────────────────────
# Node 2 – Python Interpreter
# ──────────────────────────────────────────────────────────────────────────────
_CODE_GEN_SYSTEM = """\
You are a financial data scientist specialising in Indian company audit analytics.
Write a self-contained Python script that:
  1. Scans the document text carefully for EVERY rupee / crore / lakh / percentage figure.
  2. Assigns each found figure to a clearly named variable (e.g. revenue_fy25, ppe_gross).
  3. Computes ALL of these ratios if data exists (print "N/A — not found" if missing):
       current_ratio, debt_equity_ratio, net_profit_margin_pct,
       return_on_assets_pct, interest_coverage, working_capital, dscr
  4. Flags: current_ratio < 1, debt_equity > 3, negative operating cash flow.
  5. For EVERY variable found, prints: print(f"Label: INR {{value:,.2f}} Cr")
  6. If NO figures found, ALWAYS prints:
       print("WARNING: No specific financial figures extracted from this document.")

CRITICAL: Script MUST always print at least one line. Never zero output.
Constraints: Only built-in Python, math, statistics, np (numpy). No other imports.
Output ONLY raw Python code. No markdown. Under 120 lines.
"""

_CODE_GEN_USER = """\
Extract financial figures and compute audit metrics.
CARO 2020 data fields to look for: {caro_fields}

DOCUMENT TEXT (first 6000 chars):
{text}
"""

_CODE_FIX_USER = """\
The code below had an error OR zero output. Rewrite to fix.
The script MUST print at least one line — add a fallback print if no data found.

ISSUE: {error}

ORIGINAL CODE:
{code}

Output ONLY corrected Python. No fences. Under 120 lines.
"""


def _strip_fences(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


async def python_interpreter_node(state: AnalysisState) -> dict:
    if state.get("status") == "failed":
        return state

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=api_key,
        max_output_tokens=2048,
    )

    user_msg = _CODE_GEN_USER.format(
        caro_fields=_build_caro_data_fields()[:500],
        text=state["raw_text"][:6_000],
    )

    try:
        content = await _invoke_with_retry(llm, [
            SystemMessage(content=_CODE_GEN_SYSTEM),
            HumanMessage(content=user_msg),
        ])
    except Exception as exc:
        return {
            "python_code":         "",
            "calculation_results": f"Code generation failed: {exc}",
            "status":              "calculations_done",
        }

    code   = _strip_fences(content)
    result = safe_python_executor(code)

    # Treat silent success (no printed output) as a failure — retry with fix
    no_output = result["success"] and not result["output"].strip()
    if not result["success"] or no_output:
        issue = result["error"] if not result["success"] else "Code ran but produced ZERO printed output."
        try:
            fix_content = await _invoke_with_retry(llm, [
                SystemMessage(content=_CODE_GEN_SYSTEM),
                HumanMessage(content=_CODE_FIX_USER.format(error=issue, code=code)),
            ])
            code   = _strip_fences(fix_content)
            result = safe_python_executor(code)
        except Exception as exc:
            result = {"success": False, "output": "", "error": str(exc)}

    if result["success"] and result["output"].strip():
        calc_output = result["output"]
    elif result["success"]:
        calc_output = "WARNING: No financial figures could be extracted from this document section."
    else:
        calc_output = (
            f"Calculation failed.\nError: {result['error']}\n"
            f"Partial output: {result['output']}"
        )

    return {
        "python_code":         code,
        "calculation_results": calc_output,
        "status":              "calculations_done",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 3 – Auditor (CARO 2020 clause-by-clause)
# ──────────────────────────────────────────────────────────────────────────────
_AUDITOR_SYSTEM = """\
You are a Senior Statutory Auditor conducting a formal audit under the Companies
(Auditor's Report) Order, 2020 (CARO 2020) and Indian Standards on Auditing
(SA 700 / SA 705 / SA 706).

You have been provided:
  (A) Compliance rules from the regulatory knowledge base (CARO, SA standards)
  (B) The financial document content (direct extract)
  (C) Computed financial metrics from Python analysis
  (D) The complete CARO 2020 checklist — all 21 clauses with audit questions

YOUR TASK: Produce a structured CARO 2020 compliance audit report.

PART 1 — CLAUSE-BY-CLAUSE ASSESSMENT
For each Clause 01 through 21:
  Clause XX: <Title>
  Status: COMPLIANT / NON-COMPLIANT / INSUFFICIENT DATA / NOT APPLICABLE
  Basis: <cite specific figure or text from document supporting this>

PART 2 — FINDINGS SUMMARY

### CRITICAL FINDINGS  [severity: HIGH]
Material misstatements or violations. Cite CARO clause + rupee figure.

### SIGNIFICANT FINDINGS  [severity: MEDIUM]
Control weaknesses, disclosure gaps. Cite clause.

### OBSERVATIONS  [severity: LOW]
Best-practice gaps or improvement areas.

### FINANCIAL HEALTH SUMMARY
Ratios computed, flags raised, overall risk: Low / Moderate / High / Critical.

### POSITIVE OBSERVATIONS
Areas of full compliance and strong practice.

Be specific. Cite rupee figures. Reference CARO clause numbers. Under 1800 words.
"""

_AUDITOR_USER = """\
(A) COMPLIANCE RULES FROM KNOWLEDGE BASE:
{rules}

(B) DOCUMENT CONTENT:
{text}

(C) COMPUTED FINANCIAL METRICS:
{calculations}

(D) CARO 2020 FULL CHECKLIST:
{caro_checklist}
"""


async def auditor_node(state: AnalysisState) -> dict:
    if state.get("status") == "failed":
        return state

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
        max_output_tokens=4096,   # needs room for 21-clause report
    )

    user_msg = _AUDITOR_USER.format(
        rules=state["compliance_rules"][:3_000],
        text=state["raw_text"][:5_000],
        calculations=state["calculation_results"],
        caro_checklist=_build_caro_checklist()[:4_000],
    )

    try:
        content = await _invoke_with_retry(llm, [
            SystemMessage(content=_AUDITOR_SYSTEM),
            HumanMessage(content=user_msg),
        ])
    except Exception as exc:
        return {"status": "failed", "error": f"Auditor node failed: {exc}"}

    return {
        "audit_findings": [content],
        "status":         "analysis_complete",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Compile LangGraph pipeline
# ──────────────────────────────────────────────────────────────────────────────
_wf = StateGraph(AnalysisState)
_wf.add_node("librarian",  librarian_node)
_wf.add_node("calculator", python_interpreter_node)
_wf.add_node("auditor",    auditor_node)

_wf.set_entry_point("librarian")
_wf.add_edge("librarian",  "calculator")
_wf.add_edge("calculator", "auditor")
_wf.add_edge("auditor",    END)

analysis_pipeline = _wf.compile()
"""
Step 2 – RAG Analysis Agent
==============================
Three-node LangGraph pipeline that runs after document ingestion:

  librarian  →  calculator  →  auditor
      │               │            │
  Queries KG     LLM writes    Gap analysis
  for rules      Python code,  against rules +
  & standards    executes it   computed metrics

Exports
-------
- analysis_pipeline : compiled LangGraph pipeline (ainvoke-able)
- AnalysisState     : TypedDict for state hand-off
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

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

# FIX: LightRAG requires QueryParam for query configuration — a plain dict
# does not work and silently falls back to default (naive) mode.
try:
    from lightrag.base import QueryParam
except ImportError:
    # Older LightRAG versions expose it at the top level
    from lightrag import QueryParam  # type: ignore[no-redef]

# Shared helpers from Step 1  (file is named agent.py — fixed import)
from agent import (
    gemini_complete, gemini_embed,
    WORKING_DIR, EMBEDDING_DIM,
    _invoke_with_retry, _is_retryable,
    api_key,
)

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
    """
    Execute *code* in a sandboxed namespace.
    Returns {"success": bool, "output": str, "error": str}.
    """
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
            exec(code, _safe_globals())  # noqa: S102

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
# Node 1 – Librarian  (LightRAG retrieval)
# ──────────────────────────────────────────────────────────────────────────────
async def librarian_node(state: AnalysisState) -> dict:
    """
    Query the LightRAG Knowledge Graph for compliance rules, financial
    thresholds, and risk indicators using QueryParam (not a plain dict).
    """
    if state.get("status") == "failed":
        return state

    try:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gemini_complete,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=8192,
                func=gemini_embed,
            ),
        )
        await rag.initialize_storages()

        queries = [
            "What are the specific regulatory compliance requirements and standards referenced?",
            "What financial reporting thresholds, ratios, or benchmarks are mentioned?",
            "What internal control weaknesses, risk factors, or audit criteria are identified?",
        ]

        results = []
        for q in queries:
            # FIX: use QueryParam object — plain dict was silently ignored
            answer = await rag.aquery(q, param=QueryParam(mode="hybrid"))
            results.append(f"[Query: {q}]\n{answer}")

        combined = ("\n\n" + "─" * 60 + "\n\n").join(results)
        return {"compliance_rules": combined, "status": "rules_fetched"}

    except Exception as exc:
        return {"status": "failed", "error": f"Librarian error: {exc}"}


# ──────────────────────────────────────────────────────────────────────────────
# Node 2 – Python Interpreter  (financial calculations)
# ──────────────────────────────────────────────────────────────────────────────
_CODE_GEN_SYSTEM = """\
You are a financial data scientist specialising in audit analytics.
Write a self-contained Python script that:
  1. Defines named variables for EVERY numerical figure extracted from the document.
  2. Computes standard financial ratios (liquidity, profitability, leverage, coverage).
  3. Calculates period-over-period changes when multiple reporting periods are present.
  4. Flags figures that deviate from typical industry norms
     (e.g. current ratio < 1, debt/equity > 3, negative operating cash flow).
  5. Prints ALL results with clear, labelled output lines.

Constraints:
- Use ONLY: built-in Python, `math`, `statistics`, and `np` (numpy).
- Do NOT import anything else.
- Output ONLY the raw Python code – no markdown fences, no explanation.
- Keep the script under 100 lines to stay within output token limits.
"""

_CODE_GEN_USER = """\
Extract financial figures and compute audit metrics for the document below.

DOCUMENT TEXT (first 5 000 characters):
{text}
"""

_CODE_FIX_USER = """\
The code below produced an error. Rewrite it to fix the problem.

ERROR:
{error}

ORIGINAL CODE:
{code}

Output ONLY the corrected Python code – no markdown fences, no explanation.
Keep the script under 100 lines.
"""


def _strip_fences(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


async def python_interpreter_node(state: AnalysisState) -> dict:
    """
    LLM code generation + sandboxed execution.
    Uses gemini-2.5-flash (1 500 RPD free tier).
    Retries on 503 via _invoke_with_retry before falling back to error state.
    """
    if state.get("status") == "failed":
        return state

    # gemini-2.5-flash: sufficient for code gen, generous free quota
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=api_key,
        max_output_tokens=2048,  # keep within free-tier output limits
    )

    # ── Step 1: code generation (with retry) ──
    user_msg = _CODE_GEN_USER.format(text=state["raw_text"][:5_000])
    try:
        content = await _invoke_with_retry(llm, [
            SystemMessage(content=_CODE_GEN_SYSTEM),
            HumanMessage(content=user_msg),
        ])
    except Exception as exc:
        return {
            "python_code":         "",
            "calculation_results": f"Code generation failed: {exc}",
            "status":              "calculations_done",  # non-fatal — auditor continues
        }

    code = _strip_fences(content)

    # ── Step 2: first execution attempt ──
    result = safe_python_executor(code)

    # ── Step 3: one self-correction attempt on failure ──
    if not result["success"]:
        try:
            fix_content = await _invoke_with_retry(llm, [
                SystemMessage(content=_CODE_GEN_SYSTEM),
                HumanMessage(content=_CODE_FIX_USER.format(
                    error=result["error"], code=code
                )),
            ])
            code   = _strip_fences(fix_content)
            result = safe_python_executor(code)
        except Exception as exc:
            result = {"success": False, "output": "", "error": str(exc)}

    if result["success"]:
        calc_output = result["output"] or "(code ran successfully but produced no printed output)"
    else:
        calc_output = (
            f"Calculation failed after self-correction attempt.\n"
            f"Error: {result['error']}\n"
            f"Partial output: {result['output']}"
        )

    return {
        "python_code":         code,
        "calculation_results": calc_output,
        "status":              "calculations_done",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 3 – Auditor  (gap analysis)
# ──────────────────────────────────────────────────────────────────────────────
_AUDITOR_SYSTEM = """\
You are a Senior Partner-level Chartered Accountant conducting a formal financial audit.
You have:
  (A) Compliance rules and regulatory requirements from the knowledge graph.
  (B) Computed financial metrics from the Python analysis tool.
  (C) The original document content.

Produce a structured gap analysis with these mandatory sections:

### CRITICAL FINDINGS  [severity: HIGH]
Material misstatements, regulatory violations, or fraud risk indicators.
Cite specific figures and applicable standards for each finding.

### SIGNIFICANT FINDINGS  [severity: MEDIUM]
Internal control weaknesses, disclosure deficiencies, or policy deviations.

### OBSERVATIONS  [severity: LOW]
Best-practice gaps, minor procedural issues, or improvement opportunities.

### FINANCIAL ANOMALIES
Ratios outside normal ranges, unexplained variances, or suspicious trends.

### POSITIVE OBSERVATIONS
Areas of full compliance, robust controls, or noteworthy best practices.

Be concise but specific. Cite numbers. Reference the applicable rule for every point.
Keep total output under 1500 words to stay within free-tier token limits.
"""

_AUDITOR_USER = """\
COMPLIANCE RULES & REGULATORY REQUIREMENTS:
{rules}

COMPUTED FINANCIAL METRICS:
{calculations}

DOCUMENT CONTENT (first 6 000 characters):
{text}
"""


async def auditor_node(state: AnalysisState) -> dict:
    """
    Gap analysis combining retrieved rules, computed metrics, and raw text.
    Uses gemini-2.5-flash to preserve the 25 RPD gemini-2.5-pro free quota
    for the report generation stage where deeper reasoning matters most.
    """
    if state.get("status") == "failed":
        return state

    # FIX: was gemini-2.5-pro (25 RPD free limit — exhausted quickly).
    # gemini-2.5-flash handles gap analysis well within free tier.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
        max_output_tokens=3000,
    )

    user_msg = _AUDITOR_USER.format(
        rules=state["compliance_rules"][:3_000],
        calculations=state["calculation_results"],
        text=state["raw_text"][:6_000],
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

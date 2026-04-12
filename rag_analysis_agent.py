"""
Step 2 – RAG Analysis Agent
==============================
Three-node LangGraph pipeline that runs after document ingestion:

  librarian  →  calculator  →  auditor
      │               │            │
  Queries rules    LLM writes    Gap analysis
  ChromaDB +       Python code,  against rules +
  doc KG           executes it   computed metrics

Changes from v2.1
-----------------
- librarian_node now queries the persistent RULES ChromaDB store first,
  then the document-specific LightRAG KG. Rules and document knowledge
  are combined and clearly labelled for the auditor.
- All models switched to gemini-2.5-flash (pro free tier limit was 0 RPD).
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

try:
    from lightrag.base import QueryParam
except ImportError:
    from lightrag import QueryParam  # type: ignore[no-redef]

from agent import (
    gemini_complete, gemini_embed,
    WORKING_DIR, EMBEDDING_DIM,
    _invoke_with_retry, _is_retryable,
    api_key,
)

# Rules knowledge base (pre-ingested once, queried every run)
from rules_store import query_rules


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
# Node 1 – Librarian
# Queries BOTH:
#   (A) Pre-defined rules ChromaDB  — standards that apply to every audit
#   (B) Document-specific LightRAG KG — entities extracted from the PDF
# ──────────────────────────────────────────────────────────────────────────────
async def librarian_node(state: AnalysisState) -> dict:
    """
    Two-source retrieval:
    1. Rules ChromaDB  → compliance standards uploaded once by the user
    2. LightRAG KG     → entities/relationships from the current document
    """
    if state.get("status") == "failed":
        return state

    try:
        # ── Part A: Query the pre-defined rules knowledge base ──────────────
        rule_queries = [
            "regulatory compliance requirements and applicable standards for financial audit",
            "financial reporting thresholds ratios capital adequacy benchmarks",
            "internal control requirements risk assessment and audit materiality criteria",
        ]

        # Run all three rule queries concurrently
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
                rules_section += f"### Query: {q}\n{r}\n\n"

        # ── Part B: Query the document-specific LightRAG KG ─────────────────
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

        doc_queries = [
            "What financial figures, accounts, and monetary amounts are reported?",
            "What entities, subsidiaries, or related parties are mentioned?",
            "What risk factors or contingent liabilities are disclosed?",
        ]

        doc_section = "## DOCUMENT-SPECIFIC KNOWLEDGE (from ingested financial document)\n\n"
        for q in doc_queries:
            try:
                answer = await rag.aquery(q, param=QueryParam(mode="hybrid"))
                doc_section += f"### {q}\n{answer}\n\n"
            except Exception as exc:
                doc_section += f"### {q}\n[Query failed: {exc}]\n\n"

        combined = rules_section + "\n" + ("═" * 60) + "\n\n" + doc_section
        return {"compliance_rules": combined, "status": "rules_fetched"}

    except Exception as exc:
        return {"status": "failed", "error": f"Librarian error: {exc}"}


# ──────────────────────────────────────────────────────────────────────────────
# Node 2 – Python Interpreter
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
- Keep the script under 100 lines.
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
    if state.get("status") == "failed":
        return state

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=api_key,
        max_output_tokens=2048,
    )

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
            "status":              "calculations_done",
        }

    code   = _strip_fences(content)
    result = safe_python_executor(code)

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
            f"Error: {result['error']}\nPartial output: {result['output']}"
        )

    return {
        "python_code":         code,
        "calculation_results": calc_output,
        "status":              "calculations_done",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 3 – Auditor
# ──────────────────────────────────────────────────────────────────────────────
_AUDITOR_SYSTEM = """\
You are a Senior Partner-level Chartered Accountant conducting a formal financial audit.
You have:
  (A) Compliance rules from the pre-defined regulatory knowledge base.
  (B) Document-specific entities and data from the knowledge graph.
  (C) Computed financial metrics from the Python analysis tool.
  (D) The original document content.

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
Keep total output under 1500 words.
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
    if state.get("status") == "failed":
        return state

    # FIX: was gemini-2.5-pro which exhausted the free-tier 0-RPD quota.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
        max_output_tokens=3000,
    )

    user_msg = _AUDITOR_USER.format(
        rules=state["compliance_rules"][:4_000],
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
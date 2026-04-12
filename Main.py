"""
FastAPI Orchestrator
====================
Three pipeline endpoints + one retrieval endpoint:

  POST /ingest/                    → Step 1 only
  POST /analyze/                   → Steps 1 + 2
  POST /audit/                     → Steps 1 + 2 + 3  →  saves report to disk
  GET  /report/{report_id}         → retrieve any saved report by ID

Reports are stored as JSON files in ./audit_reports/ and are retrievable
by the report_id returned from POST /audit/.
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── Agent imports ─────────────────────────────────────────────────────────────
# FIX: was "from agent import" but the ingestion file is agent.py (was agent2.py)
from agent import ingestion_agent
from rag_analysis_agent import analysis_pipeline, AnalysisState
from report_agent import report_pipeline, ReportState

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Autonomous Financial Audit AI Framework",
    version="2.1.0",
    description=(
        "Three-phase agentic pipeline: document ingestion → RAG analysis "
        "with Python-powered calculations → structured audit report generation. "
        "Reports are persisted and retrievable via GET /report/{report_id}."
    ),
)

UPLOAD_DIR  = "./temp_uploads"
REPORTS_DIR = "./audit_reports"
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Report persistence helpers
# ──────────────────────────────────────────────────────────────────────────────
def _save_report(report_data: dict) -> str:
    """
    Persist a report dict to disk as JSON.
    Returns an 8-character report_id (first segment of a UUID4).
    """
    report_id = str(uuid.uuid4()).split("-")[0]   # e.g. "a3f8c21b"
    path = Path(REPORTS_DIR) / f"{report_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return report_id


def _load_report(report_id: str) -> dict | None:
    """Load a report by ID. Returns None if not found."""
    path = Path(REPORTS_DIR) / f"{report_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Response models
# ──────────────────────────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    message:               str
    text_length_processed: int
    status:                str


class AnalysisResponse(BaseModel):
    status:              str
    compliance_rules:    str
    calculation_results: str
    python_code:         str
    audit_findings:      List[str]
    error:               Optional[str] = None


class AuditResponse(BaseModel):
    report_id:           str          # use this with GET /report/{report_id}
    status:              str
    audit_opinion:       str
    executive_summary:   str
    calculation_results: str
    audit_findings:      List[str]
    final_report:        str
    error:               Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────
def _pdf_path(file: UploadFile) -> str:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    return os.path.join(UPLOAD_DIR, file.filename)


async def _save_upload(file: UploadFile, path: str) -> None:
    with open(path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)


def _cleanup(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


async def _run_ingestion(file_path: str) -> dict:
    state = await ingestion_agent.ainvoke({
        "file_path": file_path,
        "raw_text":  "",
        "status":    "started",
        "error":     "",
    })
    if state["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {state.get('error')}",
        )
    return state


async def _run_analysis(ingest_state: dict) -> dict:
    state = await analysis_pipeline.ainvoke({
        "file_path":           ingest_state["file_path"],
        "raw_text":            ingest_state["raw_text"],
        "compliance_rules":    "",
        "python_code":         "",
        "calculation_results": "",
        "audit_findings":      [],
        "status":              "ingested",
        "error":               "",
    })
    if state["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {state.get('error')}",
        )
    return state


async def _run_report(ingest_state: dict, analysis_state: dict) -> dict:
    state = await report_pipeline.ainvoke({
        "raw_text":            ingest_state["raw_text"],
        "compliance_rules":    analysis_state["compliance_rules"],
        "calculation_results": analysis_state["calculation_results"],
        "python_code":         analysis_state["python_code"],
        "audit_findings":      analysis_state["audit_findings"],
        "audit_opinion":       "",
        "executive_summary":   "",
        "detailed_report":     "",
        "final_report":        "",
        "status":              "analysis_complete",
        "error":               "",
    })
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/ingest/",
    response_model=IngestResponse,
    summary="Step 1 – Ingest document into Knowledge Graph",
    tags=["Pipeline"],
)
async def ingest_document(file: UploadFile = File(...)):
    """Upload a financial PDF. Extracts text and builds the LightRAG KG."""
    file_path = _pdf_path(file)
    await _save_upload(file, file_path)
    try:
        ingest_state = await _run_ingestion(file_path)
    finally:
        _cleanup(file_path)

    return IngestResponse(
        message="Document ingested and Knowledge Graph constructed successfully.",
        text_length_processed=len(ingest_state.get("raw_text", "")),
        status=ingest_state["status"],
    )


@app.post(
    "/analyze/",
    response_model=AnalysisResponse,
    summary="Steps 1+2 – Ingest + RAG analysis + Python calculations",
    tags=["Pipeline"],
)
async def analyze_document(file: UploadFile = File(...)):
    """Two-phase pipeline: ingest PDF → retrieve rules → compute financial metrics → gap analysis."""
    file_path = _pdf_path(file)
    await _save_upload(file, file_path)
    try:
        ingest_state   = await _run_ingestion(file_path)
        analysis_state = await _run_analysis(ingest_state)
    finally:
        _cleanup(file_path)

    return AnalysisResponse(
        status=analysis_state["status"],
        compliance_rules=analysis_state["compliance_rules"],
        calculation_results=analysis_state["calculation_results"],
        python_code=analysis_state["python_code"],
        audit_findings=analysis_state["audit_findings"],
        error=analysis_state.get("error"),
    )


@app.post(
    "/audit/",
    response_model=AuditResponse,
    summary="Steps 1+2+3 – Full end-to-end audit pipeline",
    tags=["Pipeline"],
)
async def full_audit(file: UploadFile = File(...)):
    """
    Complete three-phase autonomous audit.
    Returns a report_id you can use with GET /report/{report_id} to retrieve
    the full report at any time.
    """
    file_path = _pdf_path(file)
    await _save_upload(file, file_path)
    try:
        ingest_state   = await _run_ingestion(file_path)
        analysis_state = await _run_analysis(ingest_state)
        report_state   = await _run_report(ingest_state, analysis_state)
    finally:
        _cleanup(file_path)

    # ── Persist report to disk ─────────────────────────────────────────────
    report_payload = {
        "audit_opinion":       report_state.get("audit_opinion", ""),
        "executive_summary":   report_state.get("executive_summary", ""),
        "calculation_results": analysis_state["calculation_results"],
        "audit_findings":      analysis_state["audit_findings"],
        "final_report":        report_state.get("final_report", ""),
        "python_code":         analysis_state.get("python_code", ""),
        "compliance_rules":    analysis_state.get("compliance_rules", ""),
        "status":              report_state["status"],
        "error":               report_state.get("error"),
    }
    report_id = _save_report(report_payload)

    return AuditResponse(
        report_id=report_id,
        status=report_state["status"],
        audit_opinion=report_state.get("audit_opinion", ""),
        executive_summary=report_state.get("executive_summary", ""),
        calculation_results=analysis_state["calculation_results"],
        audit_findings=analysis_state["audit_findings"],
        final_report=report_state.get("final_report", ""),
        error=report_state.get("error"),
    )


@app.get(
    "/report/{report_id}",
    response_class=PlainTextResponse,
    summary="Retrieve a saved audit report by ID",
    tags=["Reports"],
)
async def get_stored_report(report_id: str):
    """
    Retrieve the full Markdown audit report for a completed audit.
    Use the report_id returned by POST /audit/.
    """
    data = _load_report(report_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No report found with ID '{report_id}'. "
                   "Run POST /audit/ first and use the returned report_id.",
        )
    return data.get("final_report", "(report content missing)")


@app.get(
    "/report/{report_id}/full",
    summary="Retrieve full report data (all fields) as JSON",
    tags=["Reports"],
)
async def get_stored_report_json(report_id: str):
    """
    Returns the complete report JSON including findings, calculations,
    opinion, and the final Markdown report text.
    """
    data = _load_report(report_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No report found with ID '{report_id}'.",
        )
    return data


@app.get("/reports/", summary="List all stored report IDs", tags=["Reports"])
async def list_reports():
    """Returns all stored report IDs with their on-disk paths."""
    ids = [p.stem for p in Path(REPORTS_DIR).glob("*.json")]
    return {"report_ids": ids, "count": len(ids)}


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

"""
FastAPI Orchestrator  (v2.4)
============================
Endpoints:

  POST /ingest-rules/              → Upload rules PDFs (run once, persists forever)
  GET  /rules-status/              → Check how many rule chunks are indexed

  POST /ingest/                    → Step 1 only
  POST /analyze/                   → Steps 1 + 2  (skip_ingestion=true reuses cache)
  POST /audit/                     → Steps 1+2+3  (skip_ingestion=true reuses cache)

  GET  /report/{report_id}         → Full Markdown report text
  GET  /report/{report_id}/full    → All report fields as JSON
  GET  /reports/                   → List all stored report IDs
  GET  /last-run/                  → Check cached document from last ingestion

Changes in v2.4
---------------
- skip_ingestion=true on /audit/ and /analyze/ skips Step 1 entirely, loading
  raw_text from last_run_cache.json. Same document, fresh API key → no re-parse.
- Partial save: if drafter/refiner fails mid-run, audit_findings + executive_summary
  are still written to disk with status="partial". A failed run is not wasted.
- Fast API key error detection in report_agent (_is_key_error).
- CORS enabled so the standalone frontend HTML can call the API directly.
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent import ingestion_agent
from rag_analysis_agent import analysis_pipeline, AnalysisState
from report_agent import report_pipeline, ReportState
from rules_store import ingest_rules_pdfs, rules_db_status

app = FastAPI(
    title="Autonomous Financial Audit AI Framework",
    version="2.4.0",
    description=(
        "Three-phase agentic pipeline: document ingestion → CARO 2020 RAG analysis "
        "(clause-by-clause + Python calculations) → audit report. "
        "Use skip_ingestion=true on repeat runs to skip re-parsing the same document."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR     = "./temp_uploads"
REPORTS_DIR    = "./audit_reports"
RULES_DIR      = "./rules_docs"
LAST_RUN_CACHE = "./last_run_cache.json"

for d in (UPLOAD_DIR, REPORTS_DIR, RULES_DIR):
    os.makedirs(d, exist_ok=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _save_last_run(file_name: str, raw_text: str) -> None:
    with open(LAST_RUN_CACHE, "w", encoding="utf-8") as f:
        json.dump({"file_name": file_name, "raw_text": raw_text}, f, ensure_ascii=False)


def _load_last_run() -> dict | None:
    if not Path(LAST_RUN_CACHE).exists():
        return None
    with open(LAST_RUN_CACHE, encoding="utf-8") as f:
        return json.load(f)


# ── Report persistence ────────────────────────────────────────────────────────
def _save_report(report_data: dict) -> str:
    report_id = str(uuid.uuid4()).split("-")[0]
    path = Path(REPORTS_DIR) / f"{report_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return report_id


def _load_report(report_id: str) -> dict | None:
    path = Path(REPORTS_DIR) / f"{report_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Response models ───────────────────────────────────────────────────────────
class RulesIngestResponse(BaseModel):
    status: str
    pdfs_ingested: int
    chunks_stored: int
    skipped: int
    message: str

class RulesStatusResponse(BaseModel):
    status: str
    chunks: int
    db_path: str

class IngestResponse(BaseModel):
    message: str
    text_length_processed: int
    status: str

class AnalysisResponse(BaseModel):
    status: str
    compliance_rules: str
    calculation_results: str
    python_code: str
    audit_findings: List[str]
    error: Optional[str] = None

class AuditResponse(BaseModel):
    report_id: str
    status: str
    audit_opinion: str
    executive_summary: str
    calculation_results: str
    audit_findings: List[str]
    final_report: str
    skipped_ingestion: bool = False
    error: Optional[str] = None

class LastRunResponse(BaseModel):
    cached: bool
    file_name: Optional[str] = None
    text_length: Optional[int] = None


# ── Pipeline helpers ──────────────────────────────────────────────────────────
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
        "file_path": file_path, "raw_text": "", "status": "started", "error": "",
    })
    if state["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {state.get('error')}")
    return state

async def _run_analysis(ingest_state: dict) -> dict:
    state = await analysis_pipeline.ainvoke({
        "file_path":           ingest_state.get("file_path", ""),
        "raw_text":            ingest_state["raw_text"],
        "compliance_rules":    "",
        "python_code":         "",
        "calculation_results": "",
        "audit_findings":      [],
        "status":              "ingested",
        "error":               "",
    })
    if state["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Analysis failed: {state.get('error')}")
    return state

async def _run_report(ingest_state: dict, analysis_state: dict) -> dict:
    return await report_pipeline.ainvoke({
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

def _build_partial_report(analysis_state: dict, report_state: dict, error_msg: str) -> dict:
    """Save whatever we have when drafter/refiner fails — findings are never lost."""
    return {
        "audit_opinion":       report_state.get("audit_opinion", "Disclaimer"),
        "executive_summary":   report_state.get("executive_summary", ""),
        "calculation_results": analysis_state.get("calculation_results", ""),
        "audit_findings":      analysis_state.get("audit_findings", []),
        "final_report": (
            report_state.get("final_report")
            or report_state.get("detailed_report")
            or "(Report generation incomplete — see audit_findings for full CARO analysis)"
        ),
        "python_code":      analysis_state.get("python_code", ""),
        "compliance_rules": analysis_state.get("compliance_rules", ""),
        "status":           "partial",
        "error":            error_msg,
    }


# ── Rules endpoints ───────────────────────────────────────────────────────────
@app.post("/ingest-rules/", response_model=RulesIngestResponse,
          summary="Upload rules PDFs (run once)", tags=["Rules Setup"])
async def ingest_rules(files: List[UploadFile] = File(...)):
    saved_paths = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        dest = os.path.join(RULES_DIR, file.filename)
        with open(dest, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        saved_paths.append(dest)
    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid PDF files provided.")
    result = await ingest_rules_pdfs(saved_paths)
    return RulesIngestResponse(
        status=result["status"], pdfs_ingested=result["pdfs_ingested"],
        chunks_stored=result["chunks_stored"], skipped=result["skipped"],
        message=f"Rules ingested. {result['chunks_stored']} chunks from {result['pdfs_ingested']} PDF(s).",
    )

@app.get("/rules-status/", response_model=RulesStatusResponse,
         summary="Rules KB status", tags=["Rules Setup"])
async def rules_status_endpoint():
    return RulesStatusResponse(**rules_db_status())


# ── Pipeline endpoints ────────────────────────────────────────────────────────
@app.post("/ingest/", response_model=IngestResponse,
          summary="Step 1 – Parse PDF + build KG", tags=["Pipeline"])
async def ingest_document(file: UploadFile = File(...)):
    file_path = _pdf_path(file)
    await _save_upload(file, file_path)
    try:
        ingest_state = await _run_ingestion(file_path)
        _save_last_run(file.filename, ingest_state.get("raw_text", ""))
    finally:
        _cleanup(file_path)
    return IngestResponse(
        message="Document ingested and cached. Use skip_ingestion=true on future runs.",
        text_length_processed=len(ingest_state.get("raw_text", "")),
        status=ingest_state["status"],
    )


@app.post("/analyze/", response_model=AnalysisResponse,
          summary="Steps 1+2 – Ingest + CARO analysis", tags=["Pipeline"])
async def analyze_document(
    file: UploadFile = File(...),
    skip_ingestion: bool = Form(False),
):
    """skip_ingestion=true reuses cached raw text — skips PDF parsing entirely."""
    if skip_ingestion:
        cache = _load_last_run()
        if not cache:
            raise HTTPException(status_code=400, detail="No cached doc. Run without skip_ingestion first.")
        ingest_state = {"file_path": "", "raw_text": cache["raw_text"], "status": "kg_built"}
    else:
        file_path = _pdf_path(file)
        await _save_upload(file, file_path)
        try:
            ingest_state = await _run_ingestion(file_path)
            _save_last_run(file.filename, ingest_state.get("raw_text", ""))
        finally:
            _cleanup(file_path)

    analysis_state = await _run_analysis(ingest_state)
    return AnalysisResponse(
        status=analysis_state["status"],
        compliance_rules=analysis_state["compliance_rules"],
        calculation_results=analysis_state["calculation_results"],
        python_code=analysis_state["python_code"],
        audit_findings=analysis_state["audit_findings"],
        error=analysis_state.get("error"),
    )


@app.post("/audit/", response_model=AuditResponse,
          summary="Steps 1+2+3 – Full audit pipeline", tags=["Pipeline"])
async def full_audit(
    file: UploadFile = File(...),
    skip_ingestion: bool = Form(False),
):
    """
    Complete autonomous audit. Returns report_id even on partial failure.

    skip_ingestion=true → skip PDF parsing, reuse cached text from last run.
    Use this when retrying after an API key expiry: your findings won't be lost.
    """
    skipped = False

    if skip_ingestion:
        cache = _load_last_run()
        if not cache:
            raise HTTPException(status_code=400, detail="No cached document. Run once without skip_ingestion.")
        ingest_state = {"file_path": "", "raw_text": cache["raw_text"], "status": "kg_built"}
        skipped = True
    else:
        file_path = _pdf_path(file)
        await _save_upload(file, file_path)
        try:
            ingest_state = await _run_ingestion(file_path)
            _save_last_run(file.filename, ingest_state.get("raw_text", ""))
        finally:
            _cleanup(file_path)

    analysis_state = await _run_analysis(ingest_state)
    report_state   = await _run_report(ingest_state, analysis_state)

    error_msg = report_state.get("error", "")
    if report_state.get("status") in ("failed", None) or error_msg:
        report_payload = _build_partial_report(analysis_state, report_state, error_msg)
    else:
        report_payload = {
            "audit_opinion":       report_state.get("audit_opinion", ""),
            "executive_summary":   report_state.get("executive_summary", ""),
            "calculation_results": analysis_state["calculation_results"],
            "audit_findings":      analysis_state["audit_findings"],
            "final_report":        report_state.get("final_report", ""),
            "python_code":         analysis_state.get("python_code", ""),
            "compliance_rules":    analysis_state.get("compliance_rules", ""),
            "status":              report_state["status"],
            "error":               None,
        }

    report_id = _save_report(report_payload)

    return AuditResponse(
        report_id=report_id,
        status=report_payload["status"],
        audit_opinion=report_payload["audit_opinion"],
        executive_summary=report_payload["executive_summary"],
        calculation_results=report_payload["calculation_results"],
        audit_findings=report_payload["audit_findings"],
        final_report=report_payload["final_report"],
        skipped_ingestion=skipped,
        error=report_payload.get("error"),
    )


# ── Report retrieval ──────────────────────────────────────────────────────────
@app.get("/report/{report_id}", response_class=PlainTextResponse,
         summary="Get audit report (Markdown)", tags=["Reports"])
async def get_stored_report(report_id: str):
    data = _load_report(report_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"No report found: {report_id}")
    return data.get("final_report") or data.get("executive_summary", "(no report content)")

@app.get("/report/{report_id}/full", summary="Full report JSON", tags=["Reports"])
async def get_stored_report_json(report_id: str):
    data = _load_report(report_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"No report found: {report_id}")
    return data

@app.get("/reports/", summary="List all report IDs", tags=["Reports"])
async def list_reports():
    ids = [p.stem for p in Path(REPORTS_DIR).glob("*.json")]
    return {"report_ids": ids, "count": len(ids)}

@app.get("/last-run/", response_model=LastRunResponse,
         summary="Check cached document", tags=["Reports"])
async def last_run_status():
    cache = _load_last_run()
    if not cache:
        return LastRunResponse(cached=False)
    return LastRunResponse(cached=True, file_name=cache.get("file_name"),
                           text_length=len(cache.get("raw_text", "")))

@app.get("/", summary="Health check", tags=["Health"])
async def root():
    return {
        "status": "running", "version": "2.4.0",
        "rules_kb": rules_db_status(),
        "cached_doc": (_load_last_run() or {}).get("file_name"),
        "docs_url": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
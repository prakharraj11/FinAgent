"""
FastAPI Orchestrator  (v3.1 — background tasks + polling)
==========================================================
Changes from v3.0
-----------------
- Business logic extracted into _do_analysis() / _do_report() async fns.
- New non-blocking start endpoints:
    POST /session/{id}/analyze/start  → kicks off background task, returns immediately
    POST /session/{id}/report/start   → kicks off background task, returns immediately
  Frontend polls GET /session/{id} every 2s until stage changes.
- Synchronous /analyze and /report endpoints kept for backward compat.
- GET /embed-cache/status added (debugging / health).

Full endpoint list
------------------
  POST /session/                     → Upload PDF, parse text (0 API calls)
  POST /session/{id}/analyze         → Blocking: run analysis
  POST /session/{id}/analyze/start   → Non-blocking: start analysis in background
  POST /session/{id}/report          → Blocking: generate report
  POST /session/{id}/report/start    → Non-blocking: start report in background
  GET  /session/{id}                 → Poll session state (stage, analysis, report)
  GET  /sessions/                    → List recent sessions

  POST /ingest-rules/                → Upload rules PDFs (run once)
  GET  /rules-status/                → Rules KB chunk count
  GET  /embed-cache/status           → Embedding cache stats

  GET  /report/{report_id}           → Backwards-compat report retrieval
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent import ingestion_agent
from rag_analysis_agent import analysis_pipeline
from report_agent import report_pipeline
from rules_store import ingest_rules_pdfs, rules_db_status
from session_manager import (
    compute_file_hash,
    create_session,
    get_session,
    update_session,
    list_sessions,
    get_cached_analysis,
    save_cached_analysis,
)

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinAudit AI — Autonomous Financial Audit Framework",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR  = "./temp_uploads"
REPORTS_DIR = "./audit_reports"
RULES_DIR   = "./rules_docs"

for d in (UPLOAD_DIR, REPORTS_DIR, RULES_DIR):
    os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Response models
# ──────────────────────────────────────────────────────────────────────────────
class SessionStartResponse(BaseModel):
    session_id:   str
    file_name:    str
    file_hash:    str
    text_length:  int
    text_preview: str
    stage:        str
    cache_hit:    bool


class AnalysisResponse(BaseModel):
    session_id:          str
    stage:               str
    cache_hit:           bool
    compliance_rules:    str
    calculation_results: str
    python_code:         str
    audit_findings:      List[str]
    error:               Optional[str] = None


class ReportResponse(BaseModel):
    session_id:        str
    stage:             str
    cache_hit:         bool
    audit_opinion:     str
    executive_summary: str
    final_report:      str
    error:             Optional[str] = None


class StartTaskResponse(BaseModel):
    session_id: str
    stage:      str
    message:    str
    cache_hit:  bool = False


class SessionSummary(BaseModel):
    id:         str
    file_name:  Optional[str]
    stage:      Optional[str]
    created_at: Optional[str]
    has_report: bool
    cache_hit:  bool = False


class RulesIngestResponse(BaseModel):
    status:        str
    pdfs_ingested: int
    chunks_stored: int
    skipped:       int
    message:       str


class RulesStatusResponse(BaseModel):
    status:  str
    chunks:  int
    db_path: str


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _cleanup(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _save_report_json(report_data: dict) -> str:
    report_id = str(uuid.uuid4()).split("-")[0]
    path = Path(REPORTS_DIR) / f"{report_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return report_id


def _load_report_json(report_id: str) -> dict | None:
    path = Path(REPORTS_DIR) / f"{report_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Core business logic — shared by sync endpoints AND background tasks
# ──────────────────────────────────────────────────────────────────────────────
async def _do_analysis(session_id: str) -> None:
    """
    Run analysis pipeline for the session, update session state.
    Never raises — writes 'error' stage on failure so frontend can detect it.
    Called both directly (sync endpoint) and as a BackgroundTask.
    """
    try:
        session = get_session(session_id)
        if session is None:
            return

        file_hash = session["file_hash"]
        raw_text  = session["ingestion"]["raw_text"]

        # ── Cache check ────────────────────────────────────────────────
        cached = get_cached_analysis(file_hash)
        if cached and cached.get("analysis"):
            update_session(session_id, {
                "stage":     "analyzed",
                "analysis":  cached["analysis"],
                "cache_hit": True,
            })
            return

        # ── Run pipeline ───────────────────────────────────────────────
        update_session(session_id, {"stage": "analyzing"})
        analysis_state = await analysis_pipeline.ainvoke({
            "file_path":           "",
            "raw_text":            raw_text,
            "compliance_rules":    "",
            "python_code":         "",
            "calculation_results": "",
            "audit_findings":      [],
            "status":              "ingested",
            "error":               "",
        })

        if analysis_state.get("status") == "failed":
            update_session(session_id, {
                "stage": "error",
                "error": analysis_state.get("error", "Analysis pipeline failed"),
            })
            return

        analysis_data = {
            "compliance_rules":    analysis_state["compliance_rules"],
            "calculation_results": analysis_state["calculation_results"],
            "python_code":         analysis_state["python_code"],
            "audit_findings":      analysis_state["audit_findings"],
        }

        update_session(session_id, {"stage": "analyzed", "analysis": analysis_data})

        # Persist so same file never re-runs
        existing_cache = get_cached_analysis(file_hash) or {}
        existing_cache["analysis"] = analysis_data
        save_cached_analysis(file_hash, existing_cache)

    except Exception as exc:
        update_session(session_id, {"stage": "error", "error": str(exc)})


async def _do_report(session_id: str) -> None:
    """
    Run report pipeline for the session, update session state.
    Never raises — writes 'error' stage on failure.
    """
    try:
        session = get_session(session_id)
        if session is None:
            return

        file_hash = session["file_hash"]
        raw_text  = session["ingestion"]["raw_text"]
        analysis  = session["analysis"]

        # ── Cache check ────────────────────────────────────────────────
        cached = get_cached_analysis(file_hash)
        if cached and cached.get("report"):
            update_session(session_id, {
                "stage":     "done",
                "report":    cached["report"],
                "cache_hit": True,
            })
            return

        # ── Run pipeline ───────────────────────────────────────────────
        update_session(session_id, {"stage": "reporting"})
        report_state = await report_pipeline.ainvoke({
            "raw_text":            raw_text,
            "compliance_rules":    analysis["compliance_rules"],
            "calculation_results": analysis["calculation_results"],
            "python_code":         analysis["python_code"],
            "audit_findings":      analysis["audit_findings"],
            "audit_opinion":       "",
            "executive_summary":   "",
            "detailed_report":     "",
            "final_report":        "",
            "status":              "analysis_complete",
            "error":               "",
        })

        error_msg   = report_state.get("error", "")
        report_data = {
            "audit_opinion":     report_state.get("audit_opinion", "Disclaimer"),
            "executive_summary": report_state.get("executive_summary", ""),
            "final_report": (
                report_state.get("final_report")
                or report_state.get("detailed_report")
                or "(Report generation incomplete — see audit_findings above)"
            ),
        }

        final_stage = "done" if not error_msg else "error"
        update_session(session_id, {
            "stage":  final_stage,
            "report": report_data,
            "error":  error_msg or None,
        })

        # Persist cache
        existing_cache = get_cached_analysis(file_hash) or {}
        existing_cache["report"] = report_data
        save_cached_analysis(file_hash, existing_cache)

        # Standalone JSON for backwards compat
        full_payload = {**report_data, **analysis, "session_id": session_id}
        _save_report_json(full_payload)

    except Exception as exc:
        update_session(session_id, {"stage": "error", "error": str(exc)})


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Upload PDF and parse text  (0 API calls)
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/session/",
    response_model=SessionStartResponse,
    summary="Start a new audit session — upload PDF, parse text",
    tags=["Session"],
)
async def start_session(file: UploadFile = File(...)):
    """
    Upload a financial PDF.
    - Reads file bytes, computes MD5 hash.
    - Checks if this exact file was previously analyzed (cache hit).
    - Parses text via PyMuPDF + OCR — NO Gemini API calls.
    - Creates a session with the raw text and returns a preview.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    file_hash  = compute_file_hash(file_bytes)

    tmp_path = os.path.join(UPLOAD_DIR, f"{file_hash}_{file.filename}")
    with open(tmp_path, "wb") as f_out:
        f_out.write(file_bytes)

    try:
        ingest_result = await ingestion_agent.ainvoke({
            "file_path": tmp_path,
            "raw_text":  "",
            "status":    "started",
            "error":     "",
        })
    finally:
        _cleanup(tmp_path)

    if ingest_result["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Parse failed: {ingest_result.get('error')}",
        )

    raw_text  = ingest_result["raw_text"]
    cache_hit = get_cached_analysis(file_hash) is not None

    session_id = create_session(file.filename, file_hash)
    update_session(session_id, {
        "stage":     "ingested",
        "ingestion": {"raw_text": raw_text, "text_length": len(raw_text)},
        "cache_hit": cache_hit,
    })

    return SessionStartResponse(
        session_id=session_id,
        file_name=file.filename,
        file_hash=file_hash,
        text_length=len(raw_text),
        text_preview=raw_text[:600],
        stage="ingested",
        cache_hit=cache_hit,
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2a — Blocking analysis (kept for backwards compat)
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/session/{session_id}/analyze",
    response_model=AnalysisResponse,
    summary="Step 2 — Run analysis (blocking, uses cache if available)",
    tags=["Session"],
)
async def analyze_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    if session.get("stage") not in ("ingested", "analyzed"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected stage 'ingested', got '{session.get('stage')}'.",
        )

    await _do_analysis(session_id)

    session = get_session(session_id)
    if session.get("stage") == "error":
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {session.get('error')}",
        )

    analysis = session["analysis"]
    return AnalysisResponse(
        session_id=session_id,
        stage="analyzed",
        cache_hit=session.get("cache_hit", False),
        compliance_rules=analysis["compliance_rules"],
        calculation_results=analysis["calculation_results"],
        python_code=analysis["python_code"],
        audit_findings=analysis["audit_findings"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2b — Non-blocking analysis start  (recommended for UI)
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/session/{session_id}/analyze/start",
    response_model=StartTaskResponse,
    summary="Step 2 — Start analysis in background, return immediately",
    tags=["Session"],
)
async def start_analyze(session_id: str, background_tasks: BackgroundTasks):
    """
    Kicks off the analysis pipeline as a background task and returns immediately.
    Poll GET /session/{id} every 2s. Stage progresses:
      ingested → analyzing → analyzed   (or → error)
    If the file is cached, the background task completes almost instantly.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    if session.get("stage") not in ("ingested", "analyzed"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected stage 'ingested', got '{session.get('stage')}'.",
        )

    # Check cache before launching task — instant feedback
    file_hash = session["file_hash"]
    cached    = get_cached_analysis(file_hash)
    if cached and cached.get("analysis"):
        update_session(session_id, {
            "stage":     "analyzed",
            "analysis":  cached["analysis"],
            "cache_hit": True,
        })
        return StartTaskResponse(
            session_id=session_id,
            stage="analyzed",
            message="Returned from cache instantly (0 API calls).",
            cache_hit=True,
        )

    update_session(session_id, {"stage": "analyzing"})
    background_tasks.add_task(_do_analysis, session_id)
    return StartTaskResponse(
        session_id=session_id,
        stage="analyzing",
        message="Analysis started. Poll GET /session/{id} for progress.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3a — Blocking report (kept for backwards compat)
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/session/{session_id}/report",
    response_model=ReportResponse,
    summary="Step 3 — Generate report (blocking, uses cache if available)",
    tags=["Session"],
)
async def generate_report_for_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    if session.get("stage") not in ("analyzed", "done"):
        raise HTTPException(
            status_code=400,
            detail="Run analysis first (/session/{id}/analyze or /start).",
        )

    await _do_report(session_id)

    session = get_session(session_id)
    if session.get("stage") == "error":
        raise HTTPException(
            status_code=500,
            detail=f"Report failed: {session.get('error')}",
        )

    r = session["report"]
    return ReportResponse(
        session_id=session_id,
        stage="done",
        cache_hit=session.get("cache_hit", False),
        audit_opinion=r["audit_opinion"],
        executive_summary=r["executive_summary"],
        final_report=r["final_report"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3b — Non-blocking report start  (recommended for UI)
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/session/{session_id}/report/start",
    response_model=StartTaskResponse,
    summary="Step 3 — Start report generation in background, return immediately",
    tags=["Session"],
)
async def start_report(session_id: str, background_tasks: BackgroundTasks):
    """
    Kicks off the report pipeline as a background task and returns immediately.
    Poll GET /session/{id} every 2s. Stage progresses:
      analyzed → reporting → done   (or → error)
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    if session.get("stage") not in ("analyzed", "done"):
        raise HTTPException(
            status_code=400,
            detail="Run analysis first (/session/{id}/analyze or /start).",
        )

    # Check cache before launching task
    file_hash = session["file_hash"]
    cached    = get_cached_analysis(file_hash)
    if cached and cached.get("report"):
        update_session(session_id, {
            "stage":     "done",
            "report":    cached["report"],
            "cache_hit": True,
        })
        return StartTaskResponse(
            session_id=session_id,
            stage="done",
            message="Returned from cache instantly (0 API calls).",
            cache_hit=True,
        )

    update_session(session_id, {"stage": "reporting"})
    background_tasks.add_task(_do_report, session_id)
    return StartTaskResponse(
        session_id=session_id,
        stage="reporting",
        message="Report generation started. Poll GET /session/{id} for progress.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Session read endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get(
    "/session/{session_id}",
    summary="Get session state (use for polling)",
    tags=["Session"],
)
async def get_session_endpoint(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    # Strip large raw_text from response — use text_preview instead
    out = {k: v for k, v in session.items() if k != "ingestion"}
    if session.get("ingestion"):
        out["text_length"]  = session["ingestion"].get("text_length", 0)
        out["text_preview"] = session["ingestion"]["raw_text"][:600]
    return out


@app.get(
    "/sessions/",
    response_model=List[SessionSummary],
    summary="List recent sessions",
    tags=["Session"],
)
async def list_sessions_endpoint(limit: int = 20):
    return list_sessions(limit=limit)


# ──────────────────────────────────────────────────────────────────────────────
# Rules endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/ingest-rules/",
    response_model=RulesIngestResponse,
    summary="Upload rules PDFs (run once)",
    tags=["Rules Setup"],
)
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
        status=result["status"],
        pdfs_ingested=result["pdfs_ingested"],
        chunks_stored=result["chunks_stored"],
        skipped=result["skipped"],
        message=f"Rules ingested. {result['chunks_stored']} chunks from {result['pdfs_ingested']} PDF(s).",
    )


@app.get(
    "/rules-status/",
    response_model=RulesStatusResponse,
    summary="Rules KB status",
    tags=["Rules Setup"],
)
async def rules_status_endpoint():
    return RulesStatusResponse(**rules_db_status())


# ──────────────────────────────────────────────────────────────────────────────
# Embedding cache stats  (debugging / health)
# ──────────────────────────────────────────────────────────────────────────────
@app.get(
    "/embed-cache/status",
    summary="Embedding cache stats",
    tags=["Health"],
)
async def embed_cache_status():
    try:
        from embedding_cache import cache_stats
        return cache_stats()
    except ImportError:
        return {"error": "embedding_cache.py not found"}


# ──────────────────────────────────────────────────────────────────────────────
# Backwards-compatible report retrieval
# ──────────────────────────────────────────────────────────────────────────────
@app.get(
    "/report/{report_id}",
    response_class=PlainTextResponse,
    summary="Retrieve saved report (Markdown)",
    tags=["Reports"],
)
async def get_stored_report(report_id: str):
    data = _load_report_json(report_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"No report found: {report_id}")
    return data.get("final_report") or data.get("executive_summary", "(no content)")


@app.get(
    "/report/{report_id}/full",
    summary="Full report JSON",
    tags=["Reports"],
)
async def get_stored_report_json(report_id: str):
    data = _load_report_json(report_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"No report found: {report_id}")
    return data


@app.get("/", summary="Health check", tags=["Health"])
async def root():
    from embedding_cache import cache_stats
    return {
        "status":        "running",
        "version":       "3.1.0",
        "rules_kb":      rules_db_status(),
        "embed_cache":   cache_stats(),
        "docs_url":      "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
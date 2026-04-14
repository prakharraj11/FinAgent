"""
FastAPI Orchestrator  (v3.0)
============================
Old endpoints (still work, backward-compatible):
  POST /ingest/          → Step 1 only
  POST /analyze/         → Steps 1+2  (skip_ingestion=true reuses cache)
  POST /audit/           → Steps 1+2+3
  GET  /report/{id}      → Markdown report
  GET  /report/{id}/full → Full JSON
  GET  /reports/         → List all report IDs
  GET  /last-run/        → Check cached doc
  POST /ingest-rules/    → Upload rules PDFs
  GET  /rules-status/    → Rules KB status

NEW — Human-in-the-loop session endpoints:
  POST /session/start          → Upload PDF, start ingestion, return session_id
  GET  /session/{sid}/stream   → SSE stream for live progress events
  GET  /session/{sid}/status   → Current session state (polling fallback)
  POST /session/{sid}/analyze  → Trigger Step 2 (after ingestion)
  POST /session/{sid}/report   → Trigger Step 3 (after analysis)
  GET  /sessions/              → List recent sessions
  GET  /report/{id}/pdf        → Download Generated PDF

Changes in v3.0
---------------
- LightRAG removed from agent.py → ingestion is now near-instant (no KG build).
- Report agent: summarizer+drafter merged (one fewer LLM call).
- Refiner disabled by default (set ENABLE_REFINER=true in .env to re-enable).
- Per-document MD5 hash cache: same file → skip analysis entirely.
- Session state persisted to ./sessions/ for reconnect recovery.
- SSE keepalive pings every 20 s to survive cloud proxy timeouts.
- Automatic PDF Generation added to reports.
"""

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from fastapi.responses import PlainTextResponse, StreamingResponse, FileResponse
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent import ingestion_agent
from rag_analysis_agent import (
    AnalysisState,
    analysis_pipeline,
    auditor_node,
    librarian_node,
    python_interpreter_node,
)
from report_agent import ReportState, drafter_node, refiner_node, report_pipeline, ENABLE_REFINER, create_pdf_report
from rules_store import ingest_rules_pdfs, rules_db_status
from session_manager import (
    compute_file_hash,
    create_session,
    get_cached_analysis,
    get_session,
    list_sessions,
    save_cached_analysis,
    update_session,
)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Autonomous Financial Audit AI",
    version="3.0.0",
    description="CARO 2020 audit pipeline with human-in-the-loop chat interface.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# All paths respect DATA_DIR env var so a Render Disk at /data works out of the box
_DATA = os.getenv("DATA_DIR", ".")
UPLOAD_DIR     = os.path.join(_DATA, "temp_uploads")
REPORTS_DIR    = os.path.join(_DATA, "audit_reports")
RULES_DIR      = os.path.join(_DATA, "rules_docs")
LAST_RUN_CACHE = os.path.join(_DATA, "last_run_cache.json")

for _d in (UPLOAD_DIR, REPORTS_DIR, RULES_DIR):
    os.makedirs(_d, exist_ok=True)

# ── In-memory SSE queues (one per active session) ──────────────────────────────
_sse_queues: Dict[str, asyncio.Queue] = {}


# ── SSE helpers ───────────────────────────────────────────────────────────────
async def _emit(session_id: str, event: dict) -> None:
    """Put an event on the session's SSE queue (no-op if no listener)."""
    q = _sse_queues.get(session_id)
    if q:
        await q.put(event)


async def _close_sse(session_id: str) -> None:
    """Send sentinel None to signal SSE stream end."""
    q = _sse_queues.get(session_id)
    if q:
        await q.put(None)


# ── Legacy cache helpers ───────────────────────────────────────────────────────
def _save_last_run(file_name: str, raw_text: str) -> None:
    with open(LAST_RUN_CACHE, "w", encoding="utf-8") as f:
        json.dump({"file_name": file_name, "raw_text": raw_text}, f, ensure_ascii=False)


def _load_last_run() -> Optional[dict]:
    if not Path(LAST_RUN_CACHE).exists():
        return None
    with open(LAST_RUN_CACHE, encoding="utf-8") as f:
        return json.load(f)


# ── Report persistence ────────────────────────────────────────────────────────
def _save_report(report_data: dict) -> str:
    report_id = str(uuid.uuid4()).split("-")[0]
    
    # 1. Save JSON representation
    path = Path(REPORTS_DIR) / f"{report_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    # 2. Generate and Save PDF representation
    pdf_path = Path(REPORTS_DIR) / f"audit_report_{report_id}.pdf"
    try:
        create_pdf_report(report_data, str(pdf_path))
    except Exception as e:
        print(f"Warning: PDF generation failed for {report_id}: {e}")
        
    return report_id


def _load_report(report_id: str):
    path = Path(REPORTS_DIR) / f"{report_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ── Pydantic models ───────────────────────────────────────────────────────────
class RulesIngestResponse(BaseModel):
    status: str; pdfs_ingested: int; chunks_stored: int; skipped: int; message: str

class RulesStatusResponse(BaseModel):
    status: str; chunks: int; db_path: str

class IngestResponse(BaseModel):
    message: str; text_length_processed: int; status: str

class AnalysisResponse(BaseModel):
    status: str; compliance_rules: str; calculation_results: str
    python_code: str; audit_findings: List[str]; error: Optional[str] = None

class AuditResponse(BaseModel):
    report_id: str; status: str; audit_opinion: str; executive_summary: str
    calculation_results: str; audit_findings: List[str]; final_report: str
    skipped_ingestion: bool = False; error: Optional[str] = None

class LastRunResponse(BaseModel):
    cached: bool; file_name: Optional[str] = None; text_length: Optional[int] = None

class SessionStartResponse(BaseModel):
    session_id: str; message: str; file_name: str

class SessionStatusResponse(BaseModel):
    id: str; stage: str; file_name: str; has_analysis: bool; has_report: bool
    report_id: Optional[str] = None; error: Optional[str] = None


# ── Legacy pipeline helpers ────────────────────────────────────────────────────
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
        "file_path": ingest_state.get("file_path", ""),
        "raw_text": ingest_state["raw_text"],
        "compliance_rules": "", "python_code": "",
        "calculation_results": "", "audit_findings": [],
        "status": "ingested", "error": "",
    })
    if state["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Analysis failed: {state.get('error')}")
    return state

async def _run_report(ingest_state: dict, analysis_state: dict) -> dict:
    return await report_pipeline.ainvoke({
        "raw_text": ingest_state["raw_text"],
        "compliance_rules": analysis_state["compliance_rules"],
        "calculation_results": analysis_state["calculation_results"],
        "python_code": analysis_state["python_code"],
        "audit_findings": analysis_state["audit_findings"],
        "audit_opinion": "", "executive_summary": "",
        "detailed_report": "", "final_report": "",
        "status": "analysis_complete", "error": "",
    })

def _build_partial_report(analysis_state: dict, report_state: dict, error_msg: str) -> dict:
    return {
        "audit_opinion": report_state.get("audit_opinion", "Disclaimer"),
        "executive_summary": report_state.get("executive_summary", ""),
        "calculation_results": analysis_state.get("calculation_results", ""),
        "audit_findings": analysis_state.get("audit_findings", []),
        "final_report": (
            report_state.get("final_report") or report_state.get("detailed_report")
            or "(Report generation incomplete — see audit_findings for full CARO analysis)"
        ),
        "python_code": analysis_state.get("python_code", ""),
        "compliance_rules": analysis_state.get("compliance_rules", ""),
        "status": "partial", "error": error_msg,
    }


# ════════════════════════════════════════════════════════════════════════════════
# SESSION BACKGROUND TASKS
# ════════════════════════════════════════════════════════════════════════════════

async def _bg_ingest(session_id: str, file_bytes: bytes, file_name: str) -> None:
    """Background: parse PDF, save raw_text to session, update stage."""
    try:
        await _emit(session_id, {
            "type": "progress", "node": "parse",
            "message": f"📄 Reading {file_name}...",
        })

        # Save temp file
        tmp_path = os.path.join(UPLOAD_DIR, f"tmp_{session_id}.pdf")
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        try:
            state = await ingestion_agent.ainvoke({
                "file_path": tmp_path, "raw_text": "", "status": "started", "error": "",
            })
        finally:
            _cleanup(tmp_path)

        if state["status"] == "failed":
            raise RuntimeError(state.get("error", "Parsing failed"))

        raw_text = state["raw_text"]
        text_len = len(raw_text)

        update_session(session_id, {
            "stage": "ingested",
            "ingestion": {"raw_text": raw_text, "text_length": text_len},
        })
        _save_last_run(file_name, raw_text)

        await _emit(session_id, {
            "type": "ingestion_complete",
            "message": (
                f"Parsed **{file_name}** — {text_len:,} characters of financial data extracted."
            ),
            "text_length": text_len,
            "file_name": file_name,
        })

    except Exception as exc:
        update_session(session_id, {"stage": "error", "error": str(exc)})
        await _emit(session_id, {"type": "error", "message": f"Ingestion failed: {exc}"})


async def _bg_analyze(session_id: str) -> None:
    """
    Background: run librarian → calculator → auditor with SSE progress events.
    Checks hash-based cache first — if hit, skips all LLM calls.
    """
    session = get_session(session_id)
    if not session:
        return

    raw_text  = session["ingestion"]["raw_text"]
    file_hash = session["file_hash"]

    try:
        update_session(session_id, {"stage": "analyzing"})

        # ── Cache hit ──────────────────────────────────────────────────────────
        cached = get_cached_analysis(file_hash)
        if cached:
            await _emit(session_id, {
                "type": "progress", "node": "cache",
                "message": "⚡ Found cached analysis for this document — replaying instantly...",
            })
            await asyncio.sleep(0.5)
            analysis = cached["analysis"]
        else:
            # ── Librarian ──────────────────────────────────────────────────────
            await _emit(session_id, {
                "type": "progress", "node": "librarian",
                "message": "🔍 Retrieving CARO 2020 standards from the rules knowledge base...",
            })
            lib_state: AnalysisState = {
                "file_path": "", "raw_text": raw_text,
                "compliance_rules": "", "python_code": "",
                "calculation_results": "", "audit_findings": [],
                "status": "ingested", "error": "",
            }
            lib_state = {**lib_state, **(await librarian_node(lib_state))}

            if lib_state.get("status") == "failed":
                raise RuntimeError(lib_state.get("error", "Librarian failed"))

            # ── Calculator ────────────────────────────────────────────────────
            await _emit(session_id, {
                "type": "progress", "node": "calculator",
                "message": "🧮 Generating Python code and computing financial ratios...",
            })
            calc_state = {**lib_state, **(await python_interpreter_node(lib_state))}

            # ── Auditor ───────────────────────────────────────────────────────
            await _emit(session_id, {
                "type": "progress", "node": "auditor",
                "message": "Running clause-by-clause CARO 2020 compliance analysis (21 clauses)...",
            })
            audit_state = {**calc_state, **(await auditor_node(calc_state))}

            if audit_state.get("status") == "failed":
                raise RuntimeError(audit_state.get("error", "Auditor failed"))

            analysis = {
                "compliance_rules":    audit_state["compliance_rules"],
                "calculation_results": audit_state["calculation_results"],
                "python_code":         audit_state["python_code"],
                "audit_findings":      audit_state["audit_findings"],
            }

            # Persist to cache
            save_cached_analysis(file_hash, {"analysis": analysis})

        # ── Count severity mentions ────────────────────────────────────────────
        findings_text = "\n".join(analysis["audit_findings"])
        high_count   = findings_text.count("[severity: HIGH]")  + findings_text.count("HIGH")
        medium_count = findings_text.count("[severity: MEDIUM]") + findings_text.count("MEDIUM")
        low_count    = findings_text.count("[severity: LOW]")   + findings_text.count("LOW")
        # Simple dedup (each label counted once above is still rough)
        high_count   = max(0, high_count // 2)
        medium_count = max(0, medium_count // 2)
        low_count    = max(0, low_count // 2)

        update_session(session_id, {"stage": "analyzed", "analysis": analysis})

        await _emit(session_id, {
            "type":    "analysis_complete",
            "message": "📊 CARO 2020 analysis complete.",
            "high_count":   high_count,
            "medium_count": medium_count,
            "low_count":    low_count,
            "findings_preview": findings_text[:800],
        })

    except Exception as exc:
        update_session(session_id, {"stage": "error", "error": str(exc)})
        await _emit(session_id, {"type": "error", "message": f"Analysis failed: {exc}"})


async def _bg_report(session_id: str) -> None:
    """Background: run drafter (+ optional refiner) with SSE progress events."""
    session = get_session(session_id)
    if not session:
        return

    raw_text = session["ingestion"]["raw_text"]
    analysis = session["analysis"]

    try:
        update_session(session_id, {"stage": "reporting"})

        await _emit(session_id, {
            "type": "progress", "node": "drafter",
            "message": "✍️  Drafting the Independent Auditor's Report...",
        })

        draft_input: ReportState = {
            "raw_text":            raw_text,
            "compliance_rules":    analysis["compliance_rules"],
            "calculation_results": analysis["calculation_results"],
            "python_code":         analysis["python_code"],
            "audit_findings":      analysis["audit_findings"],
            "audit_opinion": "", "executive_summary": "",
            "detailed_report": "", "final_report": "",
            "status": "analysis_complete", "error": "",
        }
        draft_state = {**draft_input, **(await drafter_node(draft_input))}

        if draft_state.get("status") == "failed":
            raise RuntimeError(draft_state.get("error", "Drafter failed"))

        if ENABLE_REFINER:
            await _emit(session_id, {
                "type": "progress", "node": "refiner",
                "message": "🔎 QA review pass — checking consistency and completeness...",
            })
            final_state = {**draft_state, **(await refiner_node(draft_state))}
        else:
            final_state = {
                **draft_state,
                "final_report": draft_state.get("detailed_report", ""),
                "status": "report_complete",
            }

        report_payload = {
            "audit_opinion":       final_state.get("audit_opinion", ""),
            "executive_summary":   final_state.get("executive_summary", ""),
            "calculation_results": analysis["calculation_results"],
            "audit_findings":      analysis["audit_findings"],
            "final_report":        final_state.get("final_report", ""),
            "python_code":         analysis.get("python_code", ""),
            "compliance_rules":    analysis.get("compliance_rules", ""),
            "status":              final_state.get("status", "report_complete"),
            "error":               None,
        }

        report_id = _save_report(report_payload)

        # ── Generate PDF ───────────────────────────────────────────────────────
        pdf_available = False
        try:
            await _emit(session_id, {
                "type": "progress", "node": "pdf",
                "message": "📄 Generating PDF report...",
            })
            pdf_path = Path(REPORTS_DIR) / f"audit_report_{report_id}.pdf"
            create_pdf_report(report_payload, str(pdf_path))
            pdf_available = True
        except Exception as pdf_err:
            # PDF failure is non-fatal — report is still available as Markdown
            print(f"[WARN] PDF generation failed for {report_id}: {pdf_err}")

        update_session(session_id, {
            "stage":         "done",
            "report":        report_payload,
            "report_id":     report_id,
            "pdf_available": pdf_available,
        })

        await _emit(session_id, {
            "type":          "report_complete",
            "message":       "🎉 Audit report ready.",
            "report_id":     report_id,
            "audit_opinion": final_state.get("audit_opinion", ""),
            "final_report":  final_state.get("final_report", ""),
            "pdf_available": pdf_available,
        })

    except Exception as exc:
        update_session(session_id, {"stage": "error", "error": str(exc)})
        await _emit(session_id, {"type": "error", "message": f"Report generation failed: {exc}"})
    finally:
        # Give the frontend 2 s to receive the last event, then close stream
        await asyncio.sleep(2)
        await _close_sse(session_id)


# ════════════════════════════════════════════════════════════════════════════════
# SESSION ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/session/start", response_model=SessionStartResponse, tags=["Session"])
async def session_start(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a financial PDF to start a new human-in-the-loop audit session.
    Returns a session_id. Open /session/{id}/stream for live progress events.
    Ingestion runs in the background (near-instant now — no LightRAG KG build).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    file_hash  = compute_file_hash(file_bytes)
    session_id = create_session(file.filename, file_hash)

    background_tasks.add_task(_bg_ingest, session_id, file_bytes, file.filename)

    return SessionStartResponse(
        session_id=session_id,
        message="Session created. Open /session/{id}/stream for live updates.",
        file_name=file.filename,
    )


@app.get("/session/{session_id}/stream", tags=["Session"])
async def session_stream(session_id: str):
    """
    SSE endpoint — streams live progress events for this session.
    Open this in an EventSource BEFORE triggering analyze/report.
    Sends keepalive pings every 20 s to survive cloud proxy timeouts.

    Event types: progress | ingestion_complete | analysis_complete |
                 report_complete | error | ping | reconnect_state
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    queue: asyncio.Queue = asyncio.Queue()
    _sse_queues[session_id] = queue

    # On reconnect, immediately replay current state so frontend catches up
    stage = session.get("stage", "created")
    if stage == "ingested":
        ingestion = session.get("ingestion") or {}
        await queue.put({
            "type": "reconnect_state", "stage": "ingested",
            "message": f"Document already ingested ({ingestion.get('text_length', 0):,} chars).",
            "text_length": ingestion.get("text_length", 0),
            "file_name": session.get("file_name"),
        })
    elif stage in ("analyzed", "analyzing"):
        await queue.put({
            "type": "reconnect_state", "stage": stage,
            "message": "📊 Analysis already complete." if stage == "analyzed" else "⏳ Analysis in progress...",
        })
    elif stage in ("done", "reporting"):
        report_id = session.get("report_id")
        report    = session.get("report") or {}
        await queue.put({
            "type": "reconnect_state", "stage": stage,
            "message": "🎉 Report already generated." if stage == "done" else "⏳ Report in progress...",
            "report_id":    report_id,
            "final_report": report.get("final_report", ""),
        })

    async def event_generator():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    # Keepalive ping — prevents cloud proxy from closing idle connection
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    continue

                if msg is None:
                    yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
                    break

                yield f"data: {json.dumps(msg)}\n\n"
        finally:
            _sse_queues.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@app.post("/session/{session_id}/analyze", tags=["Session"])
async def session_analyze(session_id: str, background_tasks: BackgroundTasks):
    """
    Trigger Step 2: CARO 2020 analysis. Session must be in 'ingested' stage.
    Check /session/{id}/status or listen to SSE for completion.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    if session["stage"] not in ("ingested",):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot analyze from stage '{session['stage']}'. Must be 'ingested'.",
        )
    background_tasks.add_task(_bg_analyze, session_id)
    return {"status": "started", "message": "Analysis running. Follow /session/{id}/stream for updates."}


@app.post("/session/{session_id}/report", tags=["Session"])
async def session_report(session_id: str, background_tasks: BackgroundTasks):
    """
    Trigger Step 3: Report generation. Session must be in 'analyzed' stage.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    if session["stage"] not in ("analyzed",):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot generate report from stage '{session['stage']}'. Must be 'analyzed'.",
        )
    background_tasks.add_task(_bg_report, session_id)
    return {"status": "started", "message": "Report generation running. Follow /session/{id}/stream."}


@app.get("/session/{session_id}/status", response_model=SessionStatusResponse, tags=["Session"])
async def session_status(session_id: str):
    """Polling-safe status check — use this if SSE reconnects to catch up."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return SessionStatusResponse(
        id=session["id"],
        stage=session.get("stage", "unknown"),
        file_name=session.get("file_name", ""),
        has_analysis=session.get("analysis") is not None,
        has_report=session.get("report") is not None,
        report_id=session.get("report_id"),
        error=session.get("error"),
    )


@app.get("/sessions/", tags=["Session"])
async def list_all_sessions():
    return {"sessions": list_sessions(limit=20)}


# ════════════════════════════════════════════════════════════════════════════════
# LEGACY & DOWNLOAD ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

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


@app.get("/rules-status/", response_model=RulesStatusResponse, tags=["Rules Setup"])
async def rules_status_endpoint():
    return RulesStatusResponse(**rules_db_status())


@app.post("/ingest/", response_model=IngestResponse, tags=["Legacy Pipeline"])
async def ingest_document(file: UploadFile = File(...)):
    file_path = _pdf_path(file)
    await _save_upload(file, file_path)
    try:
        ingest_state = await _run_ingestion(file_path)
        _save_last_run(file.filename, ingest_state.get("raw_text", ""))
    finally:
        _cleanup(file_path)
    return IngestResponse(
        message="Document ingested. Use skip_ingestion=true on future runs.",
        text_length_processed=len(ingest_state.get("raw_text", "")),
        status=ingest_state["status"],
    )


@app.post("/analyze/", response_model=AnalysisResponse, tags=["Legacy Pipeline"])
async def analyze_document(
    file: UploadFile = File(...),
    skip_ingestion: bool = Form(False),
):
    if skip_ingestion:
        cache = _load_last_run()
        if not cache:
            raise HTTPException(400, "No cached doc. Run without skip_ingestion first.")
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


@app.post("/audit/", response_model=AuditResponse, tags=["Legacy Pipeline"])
async def full_audit(
    file: UploadFile = File(...),
    skip_ingestion: bool = Form(False),
):
    skipped = False
    if skip_ingestion:
        cache = _load_last_run()
        if not cache:
            raise HTTPException(400, "No cached document. Run once without skip_ingestion.")
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
        report_id=report_id, status=report_payload["status"],
        audit_opinion=report_payload["audit_opinion"],
        executive_summary=report_payload["executive_summary"],
        calculation_results=report_payload["calculation_results"],
        audit_findings=report_payload["audit_findings"],
        final_report=report_payload["final_report"],
        skipped_ingestion=skipped, error=report_payload.get("error"),
    )


@app.get("/report/{report_id}", response_class=PlainTextResponse, tags=["Reports"])
async def get_stored_report(report_id: str):
    """Returns raw Markdown format"""
    data = _load_report(report_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"No report found: {report_id}")
    return data.get("final_report") or data.get("executive_summary", "(no report content)")


@app.get("/report/{report_id}/full", tags=["Reports"])
async def get_stored_report_json(report_id: str):
    """Returns Full JSON Data"""
    data = _load_report(report_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"No report found: {report_id}")
    return data


@app.get("/report/{report_id}/pdf", tags=["Reports"])
async def get_stored_report_pdf(report_id: str):
    """Returns the generated PDF file for download"""
    pdf_path = Path(REPORTS_DIR) / f"audit_report_{report_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF Report not found. Generation might have failed.")
    return FileResponse(pdf_path, media_type='application/pdf', filename=f"audit_report_{report_id}.pdf")


@app.get("/reports/", tags=["Reports"])
async def list_reports():
    ids = [p.stem for p in Path(REPORTS_DIR).glob("*.json")]
    return {"report_ids": ids, "count": len(ids)}


@app.get("/last-run/", response_model=LastRunResponse, tags=["Reports"])
async def last_run_status():
    cache = _load_last_run()
    if not cache:
        return LastRunResponse(cached=False)
    return LastRunResponse(cached=True, file_name=cache.get("file_name"),
                           text_length=len(cache.get("raw_text", "")))


@app.get("/health", tags=["UI"])
async def health_check():
    """JSON health — used by frontend status panel."""
    rules = rules_db_status()
    return {"status": "ok", "rules_kb": rules}


@app.get("/", tags=["UI"])
async def serve_ui():
    """Serves the frontend HTML."""
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

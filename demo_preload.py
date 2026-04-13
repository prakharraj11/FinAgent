"""
demo_preload.py
===============
Run this ONCE before your school presentation to pre-cache all demo PDFs.

What it does
------------
1. Runs the full pipeline (parse → analyze → report) for each PDF in ./demo_docs/
2. Saves results to ./doc_cache/ (by MD5 hash) and ./sessions/
3. During the live demo, uploading any of these PDFs will return results
   INSTANTLY from cache — zero Gemini API calls, zero wait time.

Usage
-----
    mkdir demo_docs
    # copy your demo PDFs into demo_docs/
    python demo_preload.py

    # Then start the server:
    uvicorn Main:app --host 0.0.0.0 --port 8000

    # Open index.html in Chrome and upload any demo_docs PDF — instant results.
"""

import asyncio
import hashlib
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


async def preload():
    from session_manager import (
        compute_file_hash, create_session, update_session,
        get_cached_analysis, save_cached_analysis,
    )
    from agent import ingestion_agent
    from rag_analysis_agent import analysis_pipeline
    from report_agent import report_pipeline

    demo_dir = Path("./demo_docs")
    if not demo_dir.exists():
        print("[ERROR] ./demo_docs/ folder not found.")
        print("Create it and place your demo PDFs inside, then re-run.")
        return

    pdfs = list(demo_dir.glob("*.pdf"))
    if not pdfs:
        print("[WARN] No PDFs found in ./demo_docs/")
        return

    print(f"\n── Pre-loading {len(pdfs)} demo PDF(s) ──\n")

    for pdf_path in pdfs:
        print(f"📄  {pdf_path.name}")

        file_bytes = pdf_path.read_bytes()
        file_hash  = compute_file_hash(file_bytes)

        # Check if already fully cached
        cached = get_cached_analysis(file_hash)
        if cached and cached.get("analysis") and cached.get("report"):
            print(f"    ✓ Already cached — skipping (0 API calls)\n")
            continue

        # ── Stage 1: Parse ────────────────────────────────────────────────
        print("    → Parsing text...")
        tmp = f"./temp_uploads/__demo_{pdf_path.name}"
        Path(tmp).write_bytes(file_bytes)

        ingest_result = await ingestion_agent.ainvoke({
            "file_path": tmp, "raw_text": "", "status": "started", "error": "",
        })
        Path(tmp).unlink(missing_ok=True)

        if ingest_result["status"] == "failed":
            print(f"    ✗ Parse failed: {ingest_result.get('error')}\n")
            continue

        raw_text = ingest_result["raw_text"]
        print(f"    ✓ Parsed {len(raw_text):,} chars")

        # ── Stage 2: Analyse ──────────────────────────────────────────────
        if not (cached and cached.get("analysis")):
            print("    → Running analysis (CARO 2020 + Python metrics)...")
            analysis_state = await analysis_pipeline.ainvoke({
                "file_path": "", "raw_text": raw_text,
                "compliance_rules": "", "python_code": "",
                "calculation_results": "", "audit_findings": [],
                "status": "ingested", "error": "",
            })
            if analysis_state.get("status") == "failed":
                print(f"    ✗ Analysis failed: {analysis_state.get('error')}\n")
                continue

            analysis_data = {
                "compliance_rules":    analysis_state["compliance_rules"],
                "calculation_results": analysis_state["calculation_results"],
                "python_code":         analysis_state["python_code"],
                "audit_findings":      analysis_state["audit_findings"],
            }
            cached = cached or {}
            cached["analysis"] = analysis_data
            save_cached_analysis(file_hash, cached)
            print("    ✓ Analysis cached")
        else:
            analysis_data = cached["analysis"]
            print("    ✓ Analysis already cached")

        # ── Stage 3: Report ───────────────────────────────────────────────
        print("    → Generating report...")
        report_state = await report_pipeline.ainvoke({
            "raw_text": raw_text, **analysis_data,
            "audit_opinion": "", "executive_summary": "",
            "detailed_report": "", "final_report": "",
            "status": "analysis_complete", "error": "",
        })

        report_data = {
            "audit_opinion":     report_state.get("audit_opinion", "Disclaimer"),
            "executive_summary": report_state.get("executive_summary", ""),
            "final_report": (
                report_state.get("final_report")
                or report_state.get("detailed_report")
                or "(Incomplete)"
            ),
        }

        cached["report"] = report_data
        save_cached_analysis(file_hash, cached)

        # Create a session so it shows in the UI's past sessions list
        session_id = create_session(pdf_path.name, file_hash)
        update_session(session_id, {
            "stage":     "done",
            "ingestion": {"raw_text": raw_text, "text_length": len(raw_text)},
            "analysis":  analysis_data,
            "report":    report_data,
            "cache_hit": False,
        })

        print(f"    ✓ Report cached  (session: {session_id})")
        print(f"    Opinion: {report_data['audit_opinion']}\n")

    print("── Pre-load complete ──────────────────────────────────")
    print("Start the server:  uvicorn Main:app --host 0.0.0.0 --port 8000")
    print("Open:              index.html in Chrome")
    print("During demo:       upload any demo_docs PDF → instant results ✓\n")


if __name__ == "__main__":
    asyncio.run(preload())
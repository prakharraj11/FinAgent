"""
session_manager.py
==================
Lightweight session state + per-document analysis cache.

Each session is a JSON file in ./sessions/.
Each cached analysis (keyed by MD5 of uploaded file bytes) is in ./doc_cache/.
Max 10 cached docs to keep storage bounded on cloud deployments.
"""

import json
import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

SESSION_DIR   = "./sessions"
DOC_CACHE_DIR = "./doc_cache"
MAX_CACHED_DOCS = 10

for _d in [SESSION_DIR, DOC_CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)


# ── Hash ──────────────────────────────────────────────────────────────────────
def compute_file_hash(content: bytes) -> str:
    """MD5 hash of the uploaded file bytes — used as cache key."""
    return hashlib.md5(content).hexdigest()


# ── Sessions ──────────────────────────────────────────────────────────────────
def create_session(file_name: str, file_hash: str) -> str:
    """Create a new session, persist it, and return the session_id."""
    session_id = str(uuid.uuid4())[:8]
    _save_session(session_id, {
        "id":         session_id,
        "file_name":  file_name,
        "file_hash":  file_hash,
        # Lifecycle: created → ingesting → ingested → analyzing → analyzed → reporting → done | error
        "stage":      "created",
        "created_at": datetime.now().isoformat(),
        "ingestion":  None,   # { raw_text, text_length }
        "analysis":   None,   # { compliance_rules, calculation_results, python_code, audit_findings }
        "report":     None,   # { audit_opinion, executive_summary, final_report }
        "error":      None,
    })
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    path = Path(SESSION_DIR) / f"{session_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_session(session_id: str, updates: dict) -> None:
    session = get_session(session_id)
    if session is None:
        return
    session.update(updates)
    session["updated_at"] = datetime.now().isoformat()
    _save_session(session_id, session)


def _save_session(session_id: str, data: dict) -> None:
    path = Path(SESSION_DIR) / f"{session_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_sessions(limit: int = 20) -> list:
    """Return recent sessions sorted by modification time (newest first)."""
    sessions = []
    files = sorted(
        Path(SESSION_DIR).glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    for p in files:
        try:
            with open(p, encoding="utf-8") as f:
                s = json.load(f)
            sessions.append({
                "id":         s["id"],
                "file_name":  s.get("file_name"),
                "stage":      s.get("stage"),
                "created_at": s.get("created_at"),
                "has_report": s.get("report") is not None,
            })
        except Exception:
            pass
    return sessions


# ── Per-document analysis cache ───────────────────────────────────────────────
def get_cached_analysis(file_hash: str) -> Optional[dict]:
    """Return cached analysis for this file hash, or None if not cached."""
    path = Path(DOC_CACHE_DIR) / f"{file_hash}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cached_analysis(file_hash: str, data: dict) -> None:
    """
    Persist analysis results for this file hash.
    Enforces MAX_CACHED_DOCS by evicting the oldest entry when full.
    """
    existing = list(Path(DOC_CACHE_DIR).glob("*.json"))
    if len(existing) >= MAX_CACHED_DOCS:
        oldest = min(existing, key=lambda p: p.stat().st_mtime)
        try:
            oldest.unlink()
        except Exception:
            pass

    path = Path(DOC_CACHE_DIR) / f"{file_hash}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
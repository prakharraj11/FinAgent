"""
embedding_cache.py
==================
Disk-based cache for Gemini embedding API calls.

Embeddings are expensive (API quota) and 100% deterministic — identical
text always produces an identical vector. This module caches them as .npy
files keyed by the first 32 hex chars of the SHA-256 of the input text.

Impact on free-tier quota
-------------------------
Without cache, every analysis run calls gemini_embed 3× for the same
fixed librarian queries ("regulatory compliance requirements…", etc.).
After the first run those 3 calls are served from disk — zero API quota.
Rules re-ingestion is also free after the first pass.

Storage overhead: one 768-dim float32 vector ≈ 3.1 KB per unique text.
"""

import hashlib
import os
from pathlib import Path

import numpy as np

CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "./embedding_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _key(text: str) -> str:
    """Stable 32-char hex key for any input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def get_cached_embedding(text: str) -> np.ndarray | None:
    """
    Return the cached embedding vector for *text*, or None if not cached.

    Parameters
    ----------
    text : str   The exact text that was embedded.

    Returns
    -------
    np.ndarray of shape (768,) with dtype float32, or None.
    """
    path = Path(CACHE_DIR) / f"{_key(text)}.npy"
    if not path.exists():
        return None
    try:
        return np.load(str(path))
    except Exception:
        # Corrupt cache entry — delete and return miss
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def save_cached_embedding(text: str, vec: np.ndarray) -> None:
    """
    Persist the embedding vector for *text* to disk.

    Non-fatal: cache write failures are silently ignored so they
    never break the main pipeline.

    Parameters
    ----------
    text : str          The exact text that was embedded.
    vec  : np.ndarray   Shape (768,) or (1, 768); will be flattened.
    """
    path = Path(CACHE_DIR) / f"{_key(text)}.npy"
    try:
        np.save(str(path), vec.flatten().astype(np.float32))
    except Exception:
        pass  # Non-fatal


def cache_stats() -> dict:
    """Return simple stats for health endpoint / debugging."""
    try:
        files = list(Path(CACHE_DIR).glob("*.npy"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "cached_vectors": len(files),
            "total_size_kb":  round(total_bytes / 1024, 1),
            "cache_dir":      CACHE_DIR,
        }
    except Exception:
        return {"cached_vectors": 0, "total_size_kb": 0, "cache_dir": CACHE_DIR}
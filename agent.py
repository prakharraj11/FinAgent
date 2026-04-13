"""
Document Ingestion Agent  (v3.0 — LightRAG removed)
=====================================================
Why LightRAG was removed
------------------------
LightRAG's KG construction called Gemini once per text chunk (15-30 LLM
calls per document) for entity/relation extraction. However, the RAG
analysis agent already noted: "LightRAG KG returns None for all document
queries in practice" — meaning the KG was never actually used.

The analysis pipeline reads raw_text directly, and the rules knowledge base
uses ChromaDB (pre-ingested, cheap). Removing LightRAG saves 70-80% of
API quota usage per run with zero loss of functionality.

What remains
------------
- PDF parsing   (PyMuPDF + pytesseract OCR fallback)
- gemini_embed  (still needed for ChromaDB rules queries)
- LLM retry helper (_invoke_with_retry)
- ingestion_agent (simple: parse → END, no KG build)
"""

import os
import asyncio
import numpy as np
from typing import TypedDict
from functools import lru_cache

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai as google_genai

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Add to your .env:\n  GEMINI_API_KEY=your_key_here"
    )

EMBEDDING_DIM = 768   # text-embedding-004 / gemini-embedding-001 output dim
WORKING_DIR = "./temp_uploads"

# ──────────────────────────────────────────────────────────────────────────────
# Shared clients (cached to avoid re-creating per request)
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_genai_client() -> google_genai.Client:
    return google_genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1beta"},
    )


@lru_cache(maxsize=1)
def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=api_key,
        max_output_tokens=2048,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Retry helper (used by all agents)
# ──────────────────────────────────────────────────────────────────────────────
def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("503", "unavailable", "rate", "quota", "resource_exhausted", "429"))


async def _invoke_with_retry(llm: ChatGoogleGenerativeAI, messages) -> str:
    last_exc = None
    for attempt in range(4):
        try:
            response = await llm.ainvoke(messages)
            return response.content
        except Exception as exc:
            if _is_retryable(exc) and attempt < 3:
                wait = 2 ** (attempt + 2)   # 4 s, 8 s, 16 s
                await asyncio.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# Embedding (still needed for ChromaDB rules queries)
# ──────────────────────────────────────────────────────────────────────────────
async def gemini_embed(texts: list[str]) -> np.ndarray:
    """
    Embed texts using gemini-embedding-001 via the google-genai SDK.
    Rate-limit friendly: embeds one at a time with 429 retry logic.
    """
    client = _get_genai_client()
    vectors = []

    if isinstance(texts, str):
        texts = [texts]

    for text in texts:
        for attempt in range(3):
            try:
                response = await client.aio.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=text,
                    config={
                        "task_type": "RETRIEVAL_DOCUMENT",
                        "output_dimensionality": 768,
                    },
                )
                vectors.append(list(response.embeddings[0].values))
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 10 * (attempt + 1)   # 10 s, 20 s
                    print(f"[Embed] Quota hit, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise

    return np.array(vectors, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# LLM wrapper (kept for compatibility — used nowhere after LightRAG removal
# but harmless to keep; rag_analysis_agent imports it)
# ──────────────────────────────────────────────────────────────────────────────
async def gemini_complete(prompt: str, **kwargs) -> str:
    llm = _get_llm()
    return await _invoke_with_retry(llm, prompt)


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
class IngestionState(TypedDict):
    file_path: str
    raw_text:  str
    status:    str
    error:     str


# ──────────────────────────────────────────────────────────────────────────────
# Node – Parse PDF
# ──────────────────────────────────────────────────────────────────────────────
def parse_document(state: IngestionState) -> dict:
    """
    Extract text from a PDF using PyMuPDF.
    Falls back to pytesseract OCR for scanned/image-only pages.
    """
    try:
        import fitz          # PyMuPDF
        import pytesseract
        from PIL import Image
        import io

        doc = fitz.open(state["file_path"])
        full_text = ""

        for page in doc:
            text = page.get_text()
            if text and text.strip():
                full_text += text
            else:
                # OCR fallback for scanned pages
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))
                full_text += pytesseract.image_to_string(image) or ""

        doc.close()

        if not full_text.strip():
            return {"status": "failed", "error": "PDF appears to be empty or unreadable."}

        return {"raw_text": full_text, "status": "parsed"}

    except Exception as exc:
        return {"status": "failed", "error": f"Parsing error: {exc}"}


# ──────────────────────────────────────────────────────────────────────────────
# Compile minimal LangGraph pipeline (parse only — no KG build)
# ──────────────────────────────────────────────────────────────────────────────
_wf = StateGraph(IngestionState)
_wf.add_node("parse", parse_document)
_wf.set_entry_point("parse")
_wf.add_edge("parse", END)

ingestion_agent = _wf.compile()
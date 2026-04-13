"""
Document Ingestion Agent (v5.0 — OpenAI Wrapper Migration)
=====================================================
- LLM Generation: Switched to OpenAI wrapper to support Groq/xAI/OpenRouter.
- Function Names: Restored standard names to fix ImportErrors in rag_analysis_agent.
- Embeddings: UNTOUCHED. Still uses Gemini (gemini-embedding-001).
"""

import os
import asyncio
import numpy as np
from typing import TypedDict
from functools import lru_cache

# LangGraph & State
from langgraph.graph import StateGraph, END

# Gemini (Embeddings Only)
from google import genai as google_genai

# OpenAI Wrapper (for Groq generation)
from openai import AsyncOpenAI

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not gemini_api_key:
    raise EnvironmentError("GEMINI_API_KEY not found. Required for embeddings.")
if not groq_api_key:
    raise EnvironmentError("GROQ_API_KEY not found. Required for LLM generation.")

EMBEDDING_DIM = 768   # gemini-embedding-001 output dim
WORKING_DIR = "./temp_uploads"

# ──────────────────────────────────────────────────────────────────────────────
# Shared Clients (Cached)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_genai_client() -> google_genai.Client:
    """Client for Embeddings only."""
    return google_genai.Client(
        api_key=gemini_api_key,
        http_options={"api_version": "v1beta"},
    )

@lru_cache(maxsize=1)
def _get_llm_client() -> AsyncOpenAI:
    """Standard OpenAI async client wrapper pointing to Groq."""
    return AsyncOpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Retry Helper
# ──────────────────────────────────────────────────────────────────────────────
def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("503", "unavailable", "rate", "quota", "resource_exhausted", "429", "too_many_requests"))

async def _invoke_with_retry(messages: list) -> str:
    """
    Standard OpenAI-compatible async wrapper with retry.
    Accepts a list of standard message dicts.
    """
    client = _get_llm_client()
    last_exc = None
    
    for attempt in range(4):
        try:
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=messages,
                temperature=0.0,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as exc:
            if _is_retryable(exc) and attempt < 3:
                wait = 2 ** (attempt + 2)
                print(f"[LLM] Rate limit hit, retrying in {wait}s...")
                await asyncio.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc

# ──────────────────────────────────────────────────────────────────────────────
# Embedding (UNCHANGED - Still uses Gemini)
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
# State & Parsing
# ──────────────────────────────────────────────────────────────────────────────
class IngestionState(TypedDict):
    file_path: str
    raw_text:  str
    status:    str
    error:     str

def parse_document(state: IngestionState) -> dict:
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
# Compile minimal LangGraph pipeline
# ──────────────────────────────────────────────────────────────────────────────
_wf = StateGraph(IngestionState)
_wf.add_node("parse", parse_document)
_wf.set_entry_point("parse")
_wf.add_edge("parse", END)
ingestion_agent = _wf.compile()
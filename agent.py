"""
Step 1 – Document Ingestion Agent
==================================
Parses a financial PDF and indexes it into a LightRAG Knowledge Graph
using Gemini as both the LLM backbone and the embedding model.

Exports
-------
- ingestion_agent   : compiled LangGraph pipeline (ainvoke-able)
- gemini_complete   : LightRAG-compatible async LLM wrapper
- gemini_embed      : LightRAG-compatible async embedding wrapper
- WORKING_DIR       : shared LightRAG storage path used by all agents
"""

import os
import asyncio
import numpy as np
from typing import TypedDict
from functools import lru_cache

from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Create a .env file with:\n  GEMINI_API_KEY=your_key_here"
    )

WORKING_DIR   = "./audit_lightrag_storage"
EMBEDDING_DIM = 768   # Gemini text-embedding-004 output dimension

# ──────────────────────────────────────────────────────────────────────────────
# Singleton LLM — reused across all LightRAG calls so we don't create a new
# client instance on every entity-extraction call during KG construction.
# Uses gemini-2.5-flash:  free tier = 1 500 req/day, 10 RPM.
# max_output_tokens capped at 2048 to stay well within free-tier output limits.
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_ingestion_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=api_key,
        max_output_tokens=2048,
    )


@lru_cache(maxsize=1)
def _get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,   # FIX: was missing, caused silent auth errors
    )


# ──────────────────────────────────────────────────────────────────────────────
# Retry decorator for Gemini 503 / rate-limit errors
# Waits 4 s → 8 s → 16 s → 32 s before giving up (4 attempts total).
# ──────────────────────────────────────────────────────────────────────────────
def _is_retryable(exc: BaseException) -> bool:
    """Retry on 503 UNAVAILABLE or any rate-limit / server error."""
    msg = str(exc).lower()
    return any(keyword in msg for keyword in ("503", "unavailable", "rate", "quota", "resource_exhausted"))


# _gemini_retry = retry(
#     retry=retry_if_exception_type(Exception) & retry_if_exception_type(Exception),
#     retry=retry_if_exception_type(Exception),
#     wait=wait_exponential(multiplier=2, min=4, max=60),
#     stop=stop_after_attempt(4),
#     reraise=True,
# )

# Simpler approach — a plain async helper used everywhere:
async def _invoke_with_retry(llm: ChatGoogleGenerativeAI, messages) -> str:
    """Invoke the LLM with exponential-backoff retry on 503 / quota errors."""
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
# LightRAG-compatible wrappers
# ──────────────────────────────────────────────────────────────────────────────
async def gemini_complete(prompt: str, **kwargs) -> str:
    """
    Async LLM wrapper for LightRAG entity/relation extraction.
    Reuses the singleton LLM instance and retries on 503.
    """
    llm = _get_ingestion_llm()
    return await _invoke_with_retry(llm, prompt)


async def gemini_embed(texts: list[str]) -> np.ndarray:
    """
    Async embedding wrapper for LightRAG vector store construction.
    google_api_key is now passed correctly to avoid silent auth failures.
    """
    embedder = _get_embedding_model()
    vectors  = await embedder.aembed_documents(texts)
    return np.array(vectors)


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
class IngestionState(TypedDict):
    file_path: str
    raw_text:  str
    status:    str
    error:     str


# ──────────────────────────────────────────────────────────────────────────────
# Node 1 – Parse PDF
# ──────────────────────────────────────────────────────────────────────────────
def parse_document(state: IngestionState) -> dict:
    """Load PDF pages and join them into a single text block."""
    try:
        loader = PyPDFLoader(state["file_path"])
        docs   = loader.load()
        full_text = "\n".join(doc.page_content for doc in docs)
        if not full_text.strip():
            return {"status": "failed", "error": "PDF appears to be empty or unreadable."}
        return {"raw_text": full_text, "status": "parsed"}
    except Exception as exc:
        return {"status": "failed", "error": f"Parsing error: {exc}"}


# ──────────────────────────────────────────────────────────────────────────────
# Node 2 – Build Knowledge Graph with LightRAG
# ──────────────────────────────────────────────────────────────────────────────
async def build_knowledge_graph(state: IngestionState) -> dict:
    """
    Insert extracted text into LightRAG.
    LightRAG automatically extracts entities, relationships, and builds
    both a vector store and a graph store inside WORKING_DIR.

    Only the first 40 000 characters are indexed to stay within free-tier
    token budgets during KG construction (each chunk calls gemini_complete).
    """
    if state.get("status") == "failed":
        return state

    try:
        os.makedirs(WORKING_DIR, exist_ok=True)

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

        # Cap at 40 000 chars to limit the number of LightRAG chunk-level LLM
        # calls — each call hits Gemini, so large documents blow the free quota.
        text_to_index = state["raw_text"][:40_000]
        await rag.ainsert(text_to_index)

        return {"status": "kg_built"}

    except Exception as exc:
        return {"status": "failed", "error": f"KG build error: {exc}"}


# ──────────────────────────────────────────────────────────────────────────────
# Compile LangGraph pipeline
# ──────────────────────────────────────────────────────────────────────────────
_wf = StateGraph(IngestionState)
_wf.add_node("parse",    parse_document)
_wf.add_node("build_kg", build_knowledge_graph)

_wf.set_entry_point("parse")
_wf.add_edge("parse",    "build_kg")
_wf.add_edge("build_kg", END)

ingestion_agent = _wf.compile()

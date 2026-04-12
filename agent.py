"""
Step 1 – Document Ingestion Agent
==================================
Parses a financial PDF and indexes it into a LightRAG Knowledge Graph
using Gemini as both the LLM backbone and the embedding model.

FIX (v2.2): gemini_embed now uses google-genai SDK with api_version='v1'
             because text-embedding-004 is NOT available on v1beta (the
             default used by langchain_google_genai ≥ 2.x).
"""

import os
import asyncio
import numpy as np
from typing import TypedDict
from functools import lru_cache

from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

# NEW: use google-genai SDK directly for embeddings (fixes v1beta 404)
from google import genai as google_genai

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

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
EMBEDDING_DIM = 768   # text-embedding-004 output dimension


# ──────────────────────────────────────────────────────────────────────────────
# google-genai client — forced to v1 API so text-embedding-004 is found.
# langchain_google_genai ≥ 2.x defaults to v1beta which does NOT support
# text-embedding-004 embedContent, causing the 404 NOT_FOUND error.
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_genai_client() -> google_genai.Client:
    return google_genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1beta"},   # ← THE FIX
    )


@lru_cache(maxsize=1)
def _get_ingestion_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=api_key,
        max_output_tokens=2048,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Retry helper — used by all agents
# ──────────────────────────────────────────────────────────────────────────────
def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("503", "unavailable", "rate", "quota", "resource_exhausted"))


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
# LightRAG-compatible wrappers
# ──────────────────────────────────────────────────────────────────────────────
async def gemini_complete(prompt: str, **kwargs) -> str:
    """Async LLM wrapper for LightRAG entity/relation extraction."""
    llm = _get_ingestion_llm()
    return await _invoke_with_retry(llm, prompt)


# async def gemini_embed(texts: list[str]) -> np.ndarray:

#     client = _get_genai_client()
#     vectors = []

#     for text in texts:
#         # Embed one at a time — safe across all SDK versions
#         response = await client.aio.models.embed_content(
#             model="models/gemini-embedding-001",
#             contents=text,
#             config={
#                     'task_type': 'RETRIEVAL_DOCUMENT',
#                     'output_dimensionality': 768
#                 }
#         )
#         # response.embeddings is a list; [0].values is the float vector
#         vectors.append(list(response.embeddings[0].values))

#     return np.array(vectors, dtype=np.float32)
async def gemini_embed(texts: list[str]) -> np.ndarray:
    client = _get_genai_client()
    vectors = []

    # Ensure we are always working with a list
    if isinstance(texts, str):
        texts = [texts]

    for text in texts:
        try:
            response = await client.aio.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text,
                config={
                    'task_type': 'RETRIEVAL_DOCUMENT',
                    'output_dimensionality': 768
                }
            )
            # Extract the raw float list from the first embedding result
            v = list(response.embeddings[0].values)
            vectors.append(v)
            
        except Exception as e:
            if "429" in str(e):
                print("Quota hit. Waiting 10s...")
                await asyncio.sleep(10)
                # Simple one-time retry
                response = await client.aio.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=text
                )
                vectors.append(list(response.embeddings[0].values))
            else:
                raise e

    return np.array(vectors, dtype=np.float32)


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
    Capped at 40 000 chars to stay within free-tier Gemini token budgets.
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
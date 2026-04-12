"""
Rules Knowledge Base (rules_store.py)
======================================
Persistent ChromaDB vector store for pre-defined audit rules and standards.

Why ChromaDB instead of LightRAG for rules?
- Rules are STATIC — no need for expensive KG construction (which calls
  Gemini once per chunk during LightRAG ingestion).
- ChromaDB stores embeddings only, queried with simple semantic search.
- Data persists across restarts via the RULES_DB_DIR folder.
- Total API cost: one embedding call per chunk, done ONCE at ingest time.

Usage
-----
One-time ingestion (CLI):
    python ingest_rules.py --rules-dir ./rules_docs/

Runtime query (called automatically by the librarian node):
    from rules_store import query_rules, rules_db_status
    results = await query_rules("regulatory capital requirements Basel III")

REST ingestion (via running server):
    POST /ingest-rules/  with multipart PDF files
"""

import os
import asyncio
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
RULES_DB_DIR    = os.getenv("RULES_DB_DIR", "./rules_chromadb")
COLLECTION_NAME = "audit_rules"
CHUNK_SIZE      = 800    # characters per chunk
CHUNK_OVERLAP   = 150    # overlap between consecutive chunks


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(RULES_DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=RULES_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
        if start >= len(text):
            break
    return chunks

def _get_or_create_collection(client: chromadb.PersistentClient):
    REQUIRED_DIM = 768 
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # Check for existing data dimensions
        existing_data = collection.peek(limit=1)
        if existing_data and existing_data.get('embeddings') and len(existing_data['embeddings']) > 0:
            existing_dim = len(existing_data['embeddings'][0])
            if existing_dim != REQUIRED_DIM:
                print(f"[DB RESET] Dimension mismatch: DB has {existing_dim}, we need {REQUIRED_DIM}. Clearing...")
                client.delete_collection(name=COLLECTION_NAME)
                # After deletion, we proceed to get_or_create
    except Exception:
        # Collection doesn't exist, which is fine
        pass 
        
    # USE get_or_create_collection INSTEAD OF create_collection
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
# def _get_or_create_collection(client: chromadb.PersistentClient):
#     return client.get_or_create_collection(
#         name=COLLECTION_NAME,
#         metadata={"hnsw:space": "cosine"},
#     )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
async def ingest_rules_pdfs(pdf_paths: List[str]) -> dict:
    """
    Load a list of PDF file paths, chunk them, embed with Gemini,
    and upsert into the persistent ChromaDB collection.

    Safe to call multiple times — uses upsert so re-running with the
    same files does NOT create duplicate chunks.

    Parameters
    ----------
    pdf_paths : list of absolute or relative file paths to PDF files

    Returns
    -------
    dict with keys: status, pdfs_ingested, chunks_stored, skipped
    """
    # Import here to avoid circular import at module load time
    from agent import gemini_embed

    if not pdf_paths:
        return {"status": "no_pdfs", "pdfs_ingested": 0, "chunks_stored": 0, "skipped": 0}

    client     = _get_chroma_client()
    collection = _get_or_create_collection(client)

    all_ids:       List[str] = []
    all_chunks:    List[str] = []
    all_metadatas: List[dict] = []
    skipped = 0

    for pdf_path in pdf_paths:
        p = Path(pdf_path)
        if not p.exists():
            skipped += 1
            continue

        try:
            loader    = PyPDFLoader(str(p))
            docs      = loader.load()
            full_text = "\n".join(d.page_content for d in docs)
        except Exception:
            skipped += 1
            continue

        if not full_text.strip():
            skipped += 1
            continue

        chunks = _chunk_text(full_text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{p.stem}__chunk_{i}"
            all_ids.append(chunk_id)
            all_chunks.append(chunk)
            all_metadatas.append({
                "source":      p.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    if not all_chunks:
        return {
            "status":        "nothing_indexed",
            "pdfs_ingested": len(pdf_paths) - skipped,
            "chunks_stored": 0,
            "skipped":       skipped,
        }

    # Embed in batches of 20 to respect rate limits
    # BATCH = 20
    # all_embeddings: List[List[float]] = []
    # for i in range(0, len(all_chunks), BATCH):
    #     batch = all_chunks[i : i + BATCH]
    #     vecs  = await gemini_embed(batch)
    #     all_embeddings.extend(vecs.tolist())
    #     if i + BATCH < len(all_chunks):
    #         await asyncio.sleep(1)   # small pause between embedding batches
    # --- MODIFIED: Super-Safe Embedding Loop for Free Tier ---
    BATCH = 5  # Smaller batch is safer
    all_embeddings: List[List[float]] = []
    
    print(f"  [DB] Embedding {len(all_chunks)} chunks...")
    
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i : i + BATCH]
        
        success = False
        while not success:
            try:
                vecs = await gemini_embed(batch)
                all_embeddings.extend(vecs.tolist())
                success = True
                # Wait 5 seconds between EVERY successful batch
                await asyncio.sleep(5) 
            except Exception as e:
                if "429" in str(e):
                    print(f"  [QUOTA] Rate limit hit. Sleeping 60s to reset...")
                    await asyncio.sleep(60) # Full minute rest
                else:
                    raise e

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )

    return {
        "status":        "ok",
        "pdfs_ingested": len(pdf_paths) - skipped,
        "chunks_stored": len(all_chunks),
        "skipped":       skipped,
    }


async def query_rules(query: str, n_results: int = 6) -> str:
    """
    Semantic search over the pre-ingested rules knowledge base.

    Returns the top matching chunks formatted as a readable string,
    including the source document name for traceability.

    If the rules DB is empty, returns an instructional message so the
    pipeline degrades gracefully rather than crashing.
    """
    from agent import gemini_embed

    client = _get_chroma_client()

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return (
            "[Rules DB is empty — no pre-defined rules have been ingested yet. "
            "POST PDF files to /ingest-rules/ to populate the rules knowledge base.]"
        )

    count = collection.count()
    if count == 0:
        return (
            "[Rules DB is empty — POST PDF files to /ingest-rules/ first.]"
        )

    query_vec = await gemini_embed([query])
    results   = collection.query(
        query_embeddings=[query_vec[0].tolist()],
        n_results=min(n_results, count),
        include=["documents", "metadatas"],
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]

    formatted: List[str] = []
    for chunk, meta in zip(chunks, metadatas):
        source = meta.get("source", "unknown")
        formatted.append(f"[Rule source: {source}]\n{chunk}")

    return "\n\n" + ("─" * 50 + "\n\n").join(formatted)


def rules_db_status() -> dict:
    """
    Lightweight check — returns how many rule chunks are stored.
    Used by the /rules-status/ health endpoint.
    """
    try:
        client     = _get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
        return {
            "status":      "ready",
            "chunks":      collection.count(),
            "db_path":     RULES_DB_DIR,
        }
    except Exception:
        return {
            "status":  "empty",
            "chunks":  0,
            "db_path": RULES_DB_DIR,
        }
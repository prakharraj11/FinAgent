"""
ingest_rules.py
===============
One-time CLI script to bulk-load all rules PDFs from a local folder
into the persistent ChromaDB knowledge base.

Run this BEFORE starting the server, or run it while the server is
stopped — both work fine. The ChromaDB database is file-based so it
persists between runs.

Usage
-----
    # Ingest all PDFs from ./rules_docs/ (default)
    python ingest_rules.py

    # Ingest from a specific folder
    python ingest_rules.py --rules-dir /path/to/your/pdfs/

    # Check current status only (no ingestion)
    python ingest_rules.py --status
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest audit rules PDFs into the persistent ChromaDB knowledge base."
    )
    parser.add_argument(
        "--rules-dir",
        default="./Rules",
        help="Folder containing your rules PDF files (default: ./rules_docs/)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current DB status and exit without ingesting.",
    )
    args = parser.parse_args()

    # Import here so .env is loaded first
    from rules_store import ingest_rules_pdfs, rules_db_status

    if args.status:
        s = rules_db_status()
        print(f"\n── Rules Knowledge Base Status ──")
        print(f"  Status : {s['status']}")
        print(f"  Chunks : {s['chunks']}")
        print(f"  Path   : {s['db_path']}")
        return

    rules_path = Path(args.rules_dir)
    if not rules_path.exists():
        print(f"[ERROR] Folder not found: {rules_path}")
        print("Create the folder and place your rules PDFs inside it, then re-run.")
        return

    pdf_files = list(rules_path.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] No PDF files found in {rules_path}/")
        print("Place your rules/standards PDF files there and re-run.")
        return

    print(f"\n── Ingesting {len(pdf_files)} PDF(s) from {rules_path}/ ──")
    
    # REMOVED: result = await ingest_rules_pdfs([str(p) for p in pdf_files])
    
    total_chunks = 0
    total_pdfs = 0

    for p in pdf_files:
        print(f" • Processing: {p.name}")
        
        # We process one file at a time to stay under the Free Tier rate limit
        result = await ingest_rules_pdfs([str(p)])
        
        if result['status'] == 'ok':
            total_chunks += result['chunks_stored']
            total_pdfs += 1
            print(f"   √ Stored {result['chunks_stored']} chunks.")
            
            # Take a 10-second breather between different PDF files
            if p != pdf_files[-1]:
                print("   [REST] Waiting 10s for quota reset...")
                await asyncio.sleep(10)
        else:
            print(f"   × Skipped: {result['status']}")

    print(f"\n── Done ──────────────────────────────────")
    print(f"  Status        : Knowledge base updated")
    print(f"  PDFs ingested : {total_pdfs}")
    print(f"  Total Chunks  : {total_chunks}")
    print()
    print("The rules knowledge base is ready. Start the server with:")
    print("  uvicorn Main:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    asyncio.run(main())


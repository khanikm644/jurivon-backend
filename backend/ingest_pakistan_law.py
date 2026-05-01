"""
JURIVON AI v2 — Pakistan Law Ingestion Pipeline
Run weekly to keep Pakistan law knowledge current.

Usage:
  python ingest_pakistan_law.py

Requirements:
  pip install openai supabase pdfplumber httpx python-dotenv

Add PDF URLs to DOCUMENTS_TO_INGEST list below.
Official sources only — no scraping of commercial sites.
"""

import asyncio
import io
import os
import hashlib
from datetime import datetime

import httpx
import pdfplumber
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client

load_dotenv()

OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ──────────────────────────────────────────────────────────
# OFFICIAL PAKISTAN GOVERNMENT SOURCES — VERIFIED LEGAL ACCESS
# Add PDF download URLs here as you find them.
# Only use official government (.gov.pk) sources.
# ──────────────────────────────────────────────────────────
DOCUMENTS_TO_INGEST = [
    # Format:
    # {
    #   "url": "https://official-source.gov.pk/document.pdf",
    #   "source": "Source name",
    #   "doc_type": "case_law|legislation|regulation|ordinance|rule|constitution",
    #   "title": "Full title of document",
    #   "court": "Court name (for case law)",
    #   "year": 2024,
    #   "citation": "Official citation e.g. 2024 SCMR 100"
    # },

    # ── CONSTITUTION ────────────────────────────────────────
    {
        "url": "https://pakistancode.gov.pk/acts/Pakistan-Constitution-1973.pdf",
        "source": "Pakistan Code",
        "doc_type": "constitution",
        "title": "Constitution of Pakistan 1973",
        "court": None,
        "year": 1973,
        "citation": "Constitution of Pakistan 1973"
    },

    # ── KEY STATUTES — Add more from pakistancode.gov.pk ────
    # Add PDF links manually as you find them on:
    # https://pakistancode.gov.pk/
    # https://punjablaws.gov.pk/
    # https://sindhlaw.gov.pk/

    # ── SUPREME COURT JUDGMENTS ──────────────────────────────
    # Find PDFs at: https://www.supremecourt.gov.pk/judgments/
    # Add them here as you collect them

    # ── FEDERAL SHARIAT COURT ───────────────────────────────
    # Find PDFs at: https://www.federalshariatcourt.gov.pk/
]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list:
    """Split text into overlapping chunks for better RAG retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # skip tiny chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def make_doc_hash(url: str, title: str) -> str:
    """Create hash to avoid re-ingesting same document."""
    return hashlib.md5(f"{url}{title}".encode()).hexdigest()[:16]


async def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from PDF bytes."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f"  PDF extraction error: {e}")
    return text.strip()


async def already_ingested(title: str) -> bool:
    """Check if document already exists in vector store."""
    try:
        result = sb.table("pakistan_law_vectors")\
            .select("id")\
            .eq("title", title)\
            .limit(1)\
            .execute()
        return len(result.data) > 0
    except:
        return False


async def ingest_document(doc: dict) -> int:
    """
    Download, extract, chunk, embed and store one document.
    Returns number of chunks inserted.
    """
    url   = doc["url"]
    title = doc["title"]

    print(f"\n→ {title}")
    print(f"  URL: {url}")

    # Skip if already in database
    if await already_ingested(title):
        print(f"  ⏭ Already ingested — skipping")
        return 0

    # Download PDF
    try:
        async with httpx.AsyncClient(
            timeout=30,
            headers={"User-Agent": "Jurivon Legal Research/2.0 (legal@jurivon.com)"}
        ) as client:
            r = await client.get(url)
            if r.status_code != 200:
                print(f"  ✗ Download failed: HTTP {r.status_code}")
                return 0
            pdf_bytes = r.content
    except Exception as e:
        print(f"  ✗ Download error: {e}")
        return 0

    # Extract text
    text = await extract_text_from_pdf(pdf_bytes)
    if len(text.strip()) < 100:
        print(f"  ✗ Too little text extracted ({len(text)} chars)")
        return 0
    print(f"  ✓ Extracted {len(text):,} characters")

    # Chunk
    chunks = chunk_text(text, chunk_size=600, overlap=100)
    chunks = chunks[:150]  # max 150 chunks per document
    print(f"  → {len(chunks)} chunks to embed")

    # Embed and store
    inserted = 0
    for i, chunk in enumerate(chunks):
        try:
            # Get embedding from OpenAI
            emb_response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk[:8000]
            )
            embedding = emb_response.data[0].embedding

            # Store in Supabase
            sb.table("pakistan_law_vectors").insert({
                "source":     doc["source"],
                "doc_type":   doc["doc_type"],
                "title":      title,
                "court":      doc.get("court"),
                "year":       doc.get("year"),
                "citation":   doc.get("citation"),
                "chunk_text": chunk,
                "embedding":  embedding,
                "created_at": datetime.utcnow().isoformat()
            }).execute()

            inserted += 1

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  ✓ {i+1}/{len(chunks)} chunks embedded")

            # Rate limit safety
            await asyncio.sleep(0.15)

        except Exception as e:
            print(f"  ⚠ Chunk {i} error: {e}")
            await asyncio.sleep(1)
            continue

    print(f"  ✓ Inserted {inserted} chunks for: {title}")
    return inserted


async def ingest_text_directly(
    text: str,
    source: str,
    doc_type: str,
    title: str,
    court: str = None,
    year: int = None,
    citation: str = None
) -> int:
    """
    Ingest text directly (for manually copied legal text).
    Use this when you cannot get a PDF download link.
    """
    if await already_ingested(title):
        print(f"⏭ Already ingested: {title}")
        return 0

    chunks = chunk_text(text, chunk_size=600, overlap=100)
    inserted = 0

    for chunk in chunks[:50]:
        try:
            emb_response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk[:8000]
            )
            embedding = emb_response.data[0].embedding

            sb.table("pakistan_law_vectors").insert({
                "source": source, "doc_type": doc_type,
                "title": title, "court": court, "year": year,
                "citation": citation, "chunk_text": chunk,
                "embedding": embedding,
                "created_at": datetime.utcnow().isoformat()
            }).execute()

            inserted += 1
            await asyncio.sleep(0.15)
        except Exception as e:
            print(f"Chunk error: {e}")
            continue

    print(f"✓ Ingested {inserted} chunks: {title}")
    return inserted


async def build_vector_index():
    """
    Build IVFFlat index after bulk ingestion.
    Run this AFTER inserting at least 1000 chunks.
    """
    print("\nBuilding vector search index...")
    try:
        sb.rpc("exec_sql", {"sql": """
            CREATE INDEX IF NOT EXISTS idx_pk_law_embedding
            ON pakistan_law_vectors
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """}).execute()
        print("✓ Vector index built")
    except Exception as e:
        print(f"Index build: {e}")
        print("Note: Build index manually in Supabase SQL Editor after inserting 1000+ chunks")


async def search_test(query: str):
    """Test the vector search is working."""
    print(f"\nTest search: '{query}'")
    try:
        emb = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        results = sb.rpc("match_pakistan_law", {
            "query_embedding": emb.data[0].embedding,
            "match_threshold": 0.5,
            "match_count": 3
        }).execute()

        if results.data:
            print(f"✓ Found {len(results.data)} results:")
            for r in results.data:
                print(f"  - {r['title']} (similarity: {r['similarity']:.3f})")
                print(f"    {r['chunk_text'][:100]}...")
        else:
            print("No results found — add more documents first")
    except Exception as e:
        print(f"Search test failed: {e}")


async def show_stats():
    """Show current database statistics."""
    try:
        total = sb.table("pakistan_law_vectors").select("id", count="exact").execute()
        print(f"\nPakistan Law Database Statistics:")
        print(f"  Total chunks: {total.count}")

        # By doc type
        types = sb.table("pakistan_law_vectors")\
            .select("doc_type")\
            .execute()
        type_counts = {}
        for r in (types.data or []):
            t = r["doc_type"] or "unknown"
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, c in type_counts.items():
            print(f"  {t}: {c} chunks")
    except Exception as e:
        print(f"Stats error: {e}")


async def main():
    print("=" * 60)
    print("JURIVON AI — Pakistan Law Ingestion Pipeline")
    print(f"Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    await show_stats()

    if not DOCUMENTS_TO_INGEST:
        print("\n⚠ No documents configured for ingestion.")
        print("Add PDF URLs to DOCUMENTS_TO_INGEST list in this file.")
        print("\nOfficial Pakistan law sources to browse:")
        print("  https://pakistancode.gov.pk/")
        print("  https://www.supremecourt.gov.pk/judgments/")
        print("  https://www.federalshariatcourt.gov.pk/")
        print("  https://punjablaws.gov.pk/")
        print("  https://sindhlaw.gov.pk/")
        return

    total_inserted = 0
    for doc in DOCUMENTS_TO_INGEST:
        n = await ingest_document(doc)
        total_inserted += n
        await asyncio.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"COMPLETE — Total chunks inserted: {total_inserted}")

    if total_inserted > 0:
        await search_test("limitation period for breach of contract")

    await show_stats()


if __name__ == "__main__":
    asyncio.run(main())

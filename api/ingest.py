"""Ingestion pipeline: corpus -> chunks -> embeddings -> pgvector.

Run as a module:  python -m api.ingest
Chunking is intentionally simple (char window + overlap). Improving it (semantic / structural
chunking) is a documented enhancement — note your choice in the README.
"""
import asyncio
from config import settings
from db import init_schema, clear, insert_chunks
from embeddings import embed
from corpora import load_corpus


def chunk_text(text: str, size: int, overlap: int):
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


async def main():
    print(f"[ingest] corpus={settings.corpus}")
    init_schema()
    clear()

    docs = list(load_corpus(settings.corpus))   # each: {source, title, content, metadata}
    print(f"[ingest] {len(docs)} documents")

    BATCH = 64
    pending_meta, pending_text = [], []

    async def flush():
        if not pending_text:
            return
        vecs = await embed(pending_text)
        insert_chunks(
            (m["source"], m["title"], m["content"], m.get("metadata", {}), v)
            for m, v in zip(pending_meta, vecs)
        )
        pending_meta.clear()
        pending_text.clear()

    total = 0
    for d in docs:
        for piece in chunk_text(d["content"], settings.chunk_size, settings.chunk_overlap):
            pending_meta.append({**d, "content": piece})
            pending_text.append(piece)
            total += 1
            if len(pending_text) >= BATCH:
                await flush()
    await flush()
    print(f"[ingest] done: {total} chunks indexed")


if __name__ == "__main__":
    asyncio.run(main())

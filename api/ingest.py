"""Ingestion pipeline: corpus -> chunks -> embeddings -> pgvector.

Run from the repo root, naming the corpora to (re)build:
  python api/ingest.py crystal_battle pokeapi
With no arguments it ingests the CORPUS from .env. Each named corpus is replaced
in place; other corpora in the table are untouched.

Chunking is intentionally simple (char window + overlap). Improving it (semantic / structural
chunking) is a documented enhancement — note your choice in the README.
"""
import asyncio
import sys
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


async def ingest_corpus(name: str) -> int:
    print(f"[ingest] corpus={name}")
    clear(name)

    docs = list(load_corpus(name))   # each: {source, title, content, metadata}
    print(f"[ingest] {len(docs)} documents")

    BATCH = 256
    pending_meta, pending_text = [], []
    total = 0

    async def flush():
        if not pending_text:
            return
        vecs = await embed(pending_text)
        insert_chunks(
            name,
            ((m["source"], m["title"], m["content"], m.get("metadata", {}), v)
             for m, v in zip(pending_meta, vecs)),
        )
        pending_meta.clear()
        pending_text.clear()

    for d in docs:
        for n, piece in enumerate(chunk_text(d["content"], settings.chunk_size,
                                             settings.chunk_overlap)):
            if n:  # continuation chunks keep their provenance: a mid-analysis
                   # window otherwise carries no hint of what document it is
                piece = f"{d['title']} (continued): {piece}"
            pending_meta.append({**d, "content": piece})
            pending_text.append(piece)
            total += 1
            if len(pending_text) >= BATCH:
                await flush()
    await flush()
    print(f"[ingest] done: {total} chunks indexed for {name}")
    return total


async def main(names: list[str]):
    init_schema()
    grand = 0
    for name in names:
        grand += await ingest_corpus(name)
    print(f"[ingest] all done: {grand} chunks across {len(names)} corpora")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or [settings.corpus]))

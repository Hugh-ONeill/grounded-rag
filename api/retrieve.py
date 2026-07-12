"""Retrieval. MVP (vector search) is implemented. Hybrid + rerank are your V2 — marked TODO."""
from db import connect
from embeddings import embed_one
from config import settings


async def retrieve(question: str, k: int | None = None) -> list[dict]:
    """Return top-k chunks as dicts: {source, title, content, similarity}."""
    k = k or settings.top_k
    qvec = await embed_one(question)
    with connect() as conn, conn.cursor() as cur:
        # cosine distance -> similarity = 1 - distance
        cur.execute(
            """
            SELECT source, title, content, 1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, k),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---- V2: the parts that impress. Implement these next. ----

async def retrieve_hybrid(question: str, k: int | None = None) -> list[dict]:
    """TODO: combine vector search with keyword/BM25 (Postgres full-text or rank_bm25),
    merge with Reciprocal Rank Fusion, return top-k. Write up why hybrid beats pure vector."""
    raise NotImplementedError


def rerank(question: str, passages: list[dict]) -> list[dict]:
    """TODO: re-score (question, passage) pairs with a cross-encoder
    (e.g. sentence-transformers ms-marco MiniLM) and reorder. Big precision win."""
    raise NotImplementedError


def passes_threshold(passages: list[dict]) -> bool:
    """Anti-hallucination gate: if nothing is similar enough, we should answer 'I don't know'."""
    return bool(passages) and passages[0]["similarity"] >= settings.min_similarity

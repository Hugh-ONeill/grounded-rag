"""Retrieval: hybrid vector + keyword search with reciprocal rank fusion.

The vector leg finds semantically similar chunks; the keyword leg catches exact
entity names ("Sitrus Berry", "Ho-Oh") that embedding similarity dilutes in
chatty questions. Both signals also feed the refusal gate, because measured
similarity bands overlap: an off-topic question with adjacent vocabulary can
out-score an on-topic question with awkward phrasing.
"""
from db import connect
from embeddings import embed_one
from config import settings

POOL = 30  # candidates per leg before fusion
RRF_K = 60

# titles weigh 'A' (heaviest): a named entity should match a doc's title, not its body
_TSV = ("(setweight(to_tsvector('english', coalesce(title,'')), 'A') || "
        "setweight(to_tsvector('english', content), 'B'))")
# OR-semantics: comparison questions name several entities, no doc contains them all
_TSQ = "websearch_to_tsquery('english', regexp_replace(%s, '\\s+', ' OR ', 'g'))"


async def retrieve(question: str, k: int | None = None, corpus: str | None = None) -> list[dict]:
    """Vector leg: top-k chunks by cosine similarity."""
    k = k or settings.top_k
    qvec = await embed_one(question)
    where = "WHERE corpus = %s" if corpus else ""
    params = [qvec] + ([corpus] if corpus else []) + [qvec, k]
    with connect() as conn, conn.cursor() as cur:
        # cosine distance -> similarity = 1 - distance
        cur.execute(
            f"""
            SELECT id, corpus, source, title, content, 1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            params,
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _keyword_leg(question: str, k: int, corpus: str | None) -> list[dict]:
    """Keyword leg: top-k chunks by weighted full-text rank."""
    where = "AND corpus = %s" if corpus else ""
    params = [question, question] + ([corpus] if corpus else []) + [question, k]
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, corpus, source, title, content,
                   ts_rank({_TSV}, {_TSQ}) AS kw_rank
            FROM chunks
            WHERE {_TSV} @@ {_TSQ} {where}
            ORDER BY ts_rank({_TSV}, {_TSQ}) DESC
            LIMIT %s
            """,
            params,
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


async def retrieve_hybrid(question: str, k: int | None = None, corpus: str | None = None) -> list[dict]:
    """Both legs, fused with reciprocal rank fusion; top-k passages win."""
    k = k or settings.top_k
    vec = await retrieve(question, k=POOL, corpus=corpus)
    kw = _keyword_leg(question, POOL, corpus)

    fused: dict[int, dict] = {}
    for rank, p in enumerate(vec):
        f = fused.setdefault(p["id"], {**p, "kw_rank": 0.0, "rrf": 0.0})
        f["rrf"] += 1 / (RRF_K + rank + 1)
    for rank, p in enumerate(kw):
        f = fused.setdefault(p["id"], {**p, "similarity": 0.0, "rrf": 0.0})
        f["kw_rank"] = float(p["kw_rank"])
        f["rrf"] += 1 / (RRF_K + rank + 1)

    return sorted(fused.values(), key=lambda f: f["rrf"], reverse=True)[:k]


def rerank(question: str, passages: list[dict]) -> list[dict]:
    """TODO (V2): re-score (question, passage) pairs with a cross-encoder
    (e.g. sentence-transformers ms-marco MiniLM) and reorder. Also the durable
    replacement for the threshold gate: refusal margins keep narrowing as the
    corpus grows, and a calibrated relevance score beats raw cosine similarity."""
    raise NotImplementedError


def passes_threshold(passages: list[dict]) -> bool:
    """Anti-hallucination gate, two signals: strong semantic similarity OR a strong
    keyword (entity-name) match among the top passages. Measured bands overlap on
    similarity alone; see the README's threshold story."""
    top = passages[:3]
    if any(p.get("similarity", 0.0) >= settings.min_similarity for p in top):
        return True
    return any(p.get("kw_rank", 0.0) >= settings.min_keyword_rank for p in top)

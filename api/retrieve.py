"""Retrieval: hybrid vector + keyword search with reciprocal rank fusion.

The vector leg finds semantically similar chunks; the keyword leg catches exact
entity names ("Sitrus Berry", "Ho-Oh") that embedding similarity dilutes in
chatty questions.
"""
from db import connect
from embeddings import embed_one
from config import settings

POOL = 60  # candidates per leg before fusion; entity families (a species'
           # pokedex/learnset/usage/item docs) crowd ranks, so cast wide
RRF_K = 60

# titles weigh 'A' (heaviest): a named entity should match a doc's title, not its body
_TSV = ("(setweight(to_tsvector('english', coalesce(title,'')), 'A') || "
        "setweight(to_tsvector('english', content), 'B'))")

# Postgres ts_rank has no IDF: a title hit on rare "garganacl" scores the same as
# a title hit on generic "move", and OR-querying every question word floods the
# leg with common-word matches. So we select the question's rarest lexemes
# ourselves, from a document-frequency table computed at first use.
_df_cache: dict[str, int] | None = None
KW_TERMS = 4       # search at most the N rarest question terms
KW_MAX_DF = 0.05   # ignore terms in >5% of docs: "move" matches 900 titles


def _lexeme_df(conn) -> dict[str, int]:
    global _df_cache
    if _df_cache is None:
        with conn.cursor() as cur:
            cur.execute(f"SELECT word, ndoc FROM ts_stat($$SELECT {_TSV} FROM chunks$$)")
            _df_cache = {w: n for w, n in cur.fetchall()}
    return _df_cache


def _rare_terms(conn, question: str) -> list[str]:
    """Stem the question, keep lexemes that exist in the corpus, return the rarest."""
    with conn.cursor() as cur:
        cur.execute("SELECT unnest(tsvector_to_array(to_tsvector('english', %s)))", (question,))
        lexemes = [row[0] for row in cur.fetchall()]
    df = _lexeme_df(conn)
    ceiling = max(2, int(KW_MAX_DF * max(df.values(), default=0)))
    rare = sorted({l for l in lexemes if l in df and df[l] <= ceiling}, key=lambda l: df[l])
    return rare[:KW_TERMS]


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
    """Keyword leg: top-k chunks by weighted full-text rank over the question's
    rarest terms (OR-semantics: comparisons name several entities, and no doc
    contains them all). Only rare terms are searched (df ceiling), which is the
    IDF that Postgres ts_rank lacks; with common terms excluded, repetition
    flooding is gone and no length normalization is needed."""
    with connect() as conn:
        terms = _rare_terms(conn, question)
        if not terms:
            return []
        tsq = " | ".join(terms)
        where = "AND corpus = %s" if corpus else ""
        params = [tsq, tsq] + ([corpus] if corpus else []) + [k]
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, corpus, source, title, content,
                       ts_rank({_TSV}, to_tsquery('english', %s)) AS kw_rank
                FROM chunks
                WHERE {_TSV} @@ to_tsquery('english', %s) {where}
                ORDER BY kw_rank DESC
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
    and reorder. Also the durable replacement for the threshold gate."""
    raise NotImplementedError


def passes_threshold(passages: list[dict]) -> bool:
    """Anti-hallucination gate, two signals: strong semantic similarity OR a strong
    keyword (entity-name) match among the top passages."""
    top = passages[:3]
    if any(p.get("similarity", 0.0) >= settings.min_similarity for p in top):
        return True
    return any(p.get("kw_rank", 0.0) >= settings.min_keyword_rank for p in top)

"""Retrieval: hybrid vector + keyword search, RRF fusion, cross-encoder rerank.

The vector leg finds semantically similar chunks; the keyword leg catches exact
entity names ("Sitrus Berry", "Ho-Oh") that embedding similarity dilutes in
chatty questions. Fused candidates are then rescored by a cross-encoder that
reads (question, passage) together, which both improves top-k precision and
gives the refusal gate a signal that stays comparable as the corpus grows.
"""
import asyncio
import re
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
_doc_count: int = 0
KW_TERMS = 4       # search at most the N rarest question terms
KW_MAX_DF = 0.05   # ignore terms in >5% of DOCUMENTS: "move" matches 900 titles.
                   # (Of documents, not of the max lexeme frequency: popular entity
                   # names appear in every teammate list and must stay searchable.)


def _lexeme_df(conn) -> dict[str, int]:
    global _df_cache, _doc_count
    if _df_cache is None:
        with conn.cursor() as cur:
            cur.execute(f"SELECT word, ndoc FROM ts_stat($$SELECT {_TSV} FROM chunks$$)")
            _df_cache = {w: n for w, n in cur.fetchall()}
            cur.execute("SELECT count(*) FROM chunks")
            _doc_count = cur.fetchone()[0]
    return _df_cache


def _rare_terms(conn, question: str) -> list[str]:
    """Stem the question, keep lexemes that exist in the corpus, return the rarest."""
    with conn.cursor() as cur:
        cur.execute("SELECT unnest(tsvector_to_array(to_tsvector('english', %s)))", (question,))
        lexemes = [row[0] for row in cur.fetchall()]
    df = _lexeme_df(conn)
    ceiling = max(2, int(KW_MAX_DF * _doc_count))
    rare = sorted({l for l in lexemes if l in df and df[l] <= ceiling}, key=lambda l: df[l])
    return rare[:KW_TERMS]


# Accent variants for phrase search: corpus text spells "Pokémon" accented while
# questions type "pokemon", and the tsvector is not unaccented, so the two are
# different lexemes. Unaccent-at-ingest is the durable fix if this map grows.
_ACCENT_VARIANTS = {"pokemon": "pokémon", "poke": "poké"}


def _phrase_queries(conn, question: str) -> list[str]:
    """Adjacent-bigram phrase queries: the fallback IDF for questions made entirely
    of common words. No unigram in "the Sleeping Pokemon" survives the df ceiling
    (sleep 2376, pokemon 2179 vs ceiling 1130), but the PHRASE 'sleep <-> pokemon'
    is highly selective. Returns tsquery strings (already-stemmed lexeme form)."""
    words = re.findall(r"[a-z0-9'’é-]+", question.lower())
    df = _lexeme_df(conn)
    out = []
    for a, b in zip(words, words[1:]):
        for va, vb in {(a, b), (_ACCENT_VARIANTS.get(a, a), _ACCENT_VARIANTS.get(b, b))}:
            with conn.cursor() as cur:
                cur.execute("SELECT phraseto_tsquery('english', %s)::text", (f"{va} {vb}",))
                tsq = cur.fetchone()[0]
            # keep only true two-lexeme phrases whose lexemes both exist in the corpus
            lexs = re.findall(r"'([^']+)'", tsq)
            if "<->" in tsq and len(lexs) == 2 and all(l in df for l in lexs) and tsq not in out:
                out.append(tsq)
    return out


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
        if terms:
            tsq = " | ".join(terms)
            tsq_expr = "to_tsquery('english', %s)"
        else:
            # every unigram is too common to search: fall back to adjacent-word
            # phrases, whose adjacency supplies the selectivity the words lack
            phrases = _phrase_queries(conn, question)
            if not phrases:
                return []
            tsq = " | ".join(f"({p})" for p in phrases)
            tsq_expr = "%s::tsquery"
        where = "AND corpus = %s" if corpus else ""
        params = [tsq, tsq] + ([corpus] if corpus else []) + [k]
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, corpus, source, title, content,
                       ts_rank({_TSV}, {tsq_expr}) AS kw_rank
                FROM chunks
                WHERE {_TSV} @@ {tsq_expr} {where}
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

    candidates = sorted(fused.values(), key=lambda f: f["rrf"], reverse=True)
    if not settings.use_reranker:
        return candidates[:k]
    # rerank the fused pool (not just top-k): the cross-encoder can rescue a
    # passage that both cheap legs underrank
    reranked = await asyncio.to_thread(rerank, question, candidates)
    # entity anchoring: the cross-encoder overvalues definitional prose ("Priority
    # § Mechanics" outranks Kingambit's own stats for "what priority move does
    # kingambit carry"), so passages whose title carries one of the question's
    # rare terms get a bonus it cannot see
    with connect() as conn:
        rare = _rare_terms(conn, question)
    if rare:
        # ordering prior only: the boost must NOT touch rerank_score, which the
        # refusal gate reads ("Point Card" must not open the gate for a question
        # about boiling points)
        def anchored(p):
            title = (p.get("title") or p.get("source") or "").lower()
            bonus = settings.title_anchor_boost if any(t in title for t in rare) else 0.0
            return p.get("rerank_score", 0.0) + bonus
        reranked.sort(key=anchored, reverse=True)
    # diversity: entities now own many documents (pokedex, learnset, usage stats,
    # several Bulbapedia sections), and top-k stuffed with one page's chunks
    # starves the answer of its other sources. Cap chunks per page, backfill if short.
    picked, overflow, per_page = [], [], {}
    for p in reranked:
        page = (p.get("source") or "").split(" §")[0].split(" — ")[0]
        if per_page.get(page, 0) < 2:
            per_page[page] = per_page.get(page, 0) + 1
            picked.append(p)
        else:
            overflow.append(p)
        if len(picked) == k:
            break
    return (picked + overflow)[:k]


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder  # heavy import, deferred
        _reranker = CrossEncoder(settings.rerank_model)
    return _reranker


def rerank(question: str, passages: list[dict]) -> list[dict]:
    """Rescore (question, passage) pairs with the cross-encoder and reorder.
    Attaches rerank_score (sigmoid, 0-1) to each passage."""
    if not passages:
        return passages
    scores = _get_reranker().predict(
        [(question, p["content"]) for p in passages],
        activation_fn=None,  # raw logits; sigmoid below for a stable 0-1 scale
    )
    import math
    for p, s in zip(passages, scores):
        p["rerank_score"] = 1 / (1 + math.exp(-float(s)))
    return sorted(passages, key=lambda p: p["rerank_score"], reverse=True)


def passes_threshold(passages: list[dict]) -> bool:
    """Anti-hallucination gate. With the reranker on, gate on its relevance score:
    it measures answers-the-question, so its margin doesn't erode as the corpus
    grows. Fallback (reranker off): strong semantic similarity OR a strong
    keyword (entity-name) match among the top passages."""
    top = passages[:3]
    if any("rerank_score" in p for p in top):
        return any(p.get("rerank_score", 0.0) >= settings.min_rerank_score for p in top)
    if any(p.get("similarity", 0.0) >= settings.min_similarity for p in top):
        return True
    return any(p.get("kw_rank", 0.0) >= settings.min_keyword_rank for p in top)

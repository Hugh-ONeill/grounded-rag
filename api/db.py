"""Postgres + pgvector storage. This boilerplate is done; the schema is the interesting bit."""
import psycopg
from psycopg.types.json import Json
from pgvector.psycopg import register_vector
from config import settings


def connect():
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def init_schema():
    """Create the extension + chunks table. Idempotent."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          BIGSERIAL PRIMARY KEY,
                corpus      TEXT NOT NULL DEFAULT 'default',  -- which adapter produced it
                source      TEXT NOT NULL,      -- citation label, e.g. "gen9ou_chaos#Heatran"
                title       TEXT,
                content     TEXT NOT NULL,
                metadata    JSONB DEFAULT '{{}}',
                embedding   vector({settings.embed_dim})
            )
        """)
        # migrate pre-multi-corpus tables in place
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS corpus TEXT NOT NULL DEFAULT 'default'")
        # ANN index for fast cosine search. ivfflat needs data first; create after first ingest
        # or use hnsw (better recall, no training):
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_corpus ON chunks (corpus)")
        # weighted full-text index for the hybrid keyword leg (titles weigh most:
        # named entities like "Sitrus Berry" should match a doc's title, not its body)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_fts ON chunks USING gin (
                (setweight(to_tsvector('english', coalesce(title,'')), 'A') ||
                 setweight(to_tsvector('english', content), 'B'))
            )
        """)
        conn.commit()


def insert_chunks(corpus, rows):
    """rows: iterable of (source, title, content, metadata_dict, embedding_list)."""
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (corpus, source, title, content, metadata, embedding) VALUES (%s,%s,%s,%s,%s,%s)",
            [(corpus, s, t, c, Json(m), e) for s, t, c, m, e in rows],
        )
        conn.commit()


def clear(corpus=None):
    """Drop one corpus's chunks (re-ingest) or everything (corpus=None)."""
    with connect() as conn, conn.cursor() as cur:
        if corpus:
            cur.execute("DELETE FROM chunks WHERE corpus = %s", (corpus,))
        else:
            cur.execute("TRUNCATE chunks RESTART IDENTITY")
        conn.commit()
